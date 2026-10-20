"""Autoregressive GPT baseline (`algo.name: ar`) for the flow-model comparison.

A plain decoder-only transformer with the flow model's block dimensions and no flow component:
hidden `model.hidden_size`, `model.n_blocks` pre-norm blocks of causal attention (RMSNorm, rotary,
QK-norm, FA4) + fused SwiGLU MLP (quack, hidden rounded to a multiple of 64 as in `model.py`), a
final RMSNorm, and a GPT-2-style TIED token embedding / LM head (one `backbone.wte.weight`; no
bias). Same padded head convention as ReLF: `model.vocab_padded_size` rows, padded slots masked -inf
via `logit_mask`.

Training: `forward(input_ids, attention_mask)` returns `{'loss'}`, the next-token cross-entropy
averaged over the predicted positions `1..L-1` of the same packed OWT windows the flow model trains
on, through the same chunked fused linear+CE (no full `(B*L, V)` logits).

Sampling: `generate_samples(num_samples, num_steps)` is ancestral sampling from BOS with a KV
cache -- `num_steps` sampled tokens after the BOS (`sampling.steps: [1023]` gives the flow model's
1024-token sequences), `sampling.temperature` (1.0 = the model's own distribution) and optional
`sampling.top_k` / `sampling.top_p`. Logits are fp32 with TF32 off, as in the flow sampler. The
returned `(B, num_steps + 1)` LongTensor includes the leading BOS, so the entropy / gen-ppl scorers
see the same kind of sequence the flow model emits.
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func
from quack.mlp import MLP as FusedMLP
from quack.rotary import apply_rotary_emb

from relf import utils
from relf.fused_ce import chunked_linear_cross_entropy
from relf.model import RMSNorm


def _rotary_tables(head_dim, length, base=10_000):
    """Return (length, head_dim/2) fp32 cos/sin tables, non-interleaved like `model.Rotary`."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(length).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    return freqs.cos(), freqs.sin()


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(dim // n_heads)
        self.k_norm = RMSNorm(dim // n_heads)
        self.norm2 = RMSNorm(dim)
        # Same fused SwiGLU width rule as `DDiTBlock` (2048 at dim 768)
        _mlp_hidden = -(-int(mlp_ratio * dim * 2 / 3) // 64) * 64
        self.mlp = FusedMLP(dim, hidden_features=_mlp_hidden, activation="swiglu", concat_layout=True)
        self.dropout = nn.Dropout(dropout)

    def _qkv(self, x, cos, sin):
        qkv = einops.rearrange(self.attn_qkv(x), 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rotary_emb(k, cos.to(k.dtype), sin.to(k.dtype))
        return q, k, v

    def forward(self, x, cos, sin):
        """Apply the block to the full sequence `x` (B, S, dim) with causal flash attention."""
        q, k, v = self._qkv(self.norm1(x), cos, sin)
        a = flash_attn_func(q, k, v, causal=True)
        a = a[0] if isinstance(a, tuple) else a
        a = einops.rearrange(a, 'b s h d -> b s (h d)')
        x = x + self.dropout(self.attn_out(a))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

    def decode(self, x, cos, sin, k_cache, v_cache, pos):
        """Apply the block to one token for one KV-cache sampling step.

        `x` (B, 1, dim) is the token at position `pos`; `cos`/`sin` are that position's rotary rows
        (1, d/2); the caches (B, L, H, d) hold positions `< pos` and receive this one. Eval mode
        only (no dropout).
        """
        q, k, v = self._qkv(self.norm1(x), cos, sin)
        k_cache[:, pos] = k[:, 0]
        v_cache[:, pos] = v[:, 0]
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_cache[:, : pos + 1].transpose(1, 2),
            v_cache[:, : pos + 1].transpose(1, 2),
        )
        a = a.transpose(1, 2).reshape(x.shape[0], 1, -1)
        x = x + self.attn_out(a)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Embedding -> blocks -> final norm; returns pre-head features (B, S, dim) in bf16.

    `forward` is the (optionally compiled) fixed-shape training path; `decode` is the eager
    per-token KV-cache path used by sampling.
    """

    def __init__(self, config, vocab_size):
        super().__init__()
        dim = config.model.hidden_size
        self.dim = dim
        self.n_heads = config.model.n_heads
        self.length = config.model.length
        self.wte = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(config.model.dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, self.n_heads, dropout=config.model.dropout) for _ in range(config.model.n_blocks)]
        )
        self.norm_final = RMSNorm(dim)
        cos, sin = _rotary_tables(dim // self.n_heads, self.length)
        self.register_buffer('rot_cos', cos, persistent=False)
        self.register_buffer('rot_sin', sin, persistent=False)

    def forward(self, input_ids):
        x = self.drop(self.wte(input_ids).to(torch.bfloat16))
        S = x.shape[1]
        cos, sin = self.rot_cos[:S], self.rot_sin[:S]
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.norm_final(x)
        return x

    def decode(self, tok, caches, pos):
        """Map `tok` (B, 1) long at position `pos` to features (B, dim) bf16."""
        x = self.wte(tok).to(torch.bfloat16)
        cos, sin = self.rot_cos[pos : pos + 1], self.rot_sin[pos : pos + 1]
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block, (k_cache, v_cache) in zip(self.blocks, caches):
                x = block.decode(x, cos, sin, k_cache, v_cache, pos)
            x = self.norm_final(x)
        return x[:, 0]


class GPT(nn.Module):
    """Decoder-only LM with the `transformers.Trainer` / `ReLFTrainer` contract."""

    def __init__(self, config, tokenizer, vocab_size=None):
        super().__init__()
        # Not `config`: `transformers.Trainer` treats `model.config` as a `PretrainedConfig`
        self.relf_config = config
        self.tokenizer = tokenizer
        if vocab_size is None:
            vocab_size = len(tokenizer)
        self.real_vocab_size = vocab_size
        padded = config.model.get('vocab_padded_size', 50304)
        self.vocab_size = self.real_vocab_size if padded is None else int(padded)
        if self.vocab_size < self.real_vocab_size:
            raise ValueError(
                f'model.vocab_padded_size={self.vocab_size} is smaller than the real vocab size '
                f'{self.real_vocab_size}'
            )
        assert config.algo.backbone == 'gpt', config.algo.backbone
        self.num_tokens = config.model.length
        self.backbone = Transformer(config, self.vocab_size)
        if self.vocab_size > self.real_vocab_size:
            logit_mask = torch.zeros(self.vocab_size)
            logit_mask[self.real_vocab_size :] = float('-inf')
        else:
            logit_mask = None
        self.register_buffer('logit_mask', logit_mask, persistent=False)
        self.initialize_weights()
        if config.model.compile:
            # Compile the fixed-shape training path only; sampling calls the blocks' eager
            # `decode` directly (dynamic KV-cache lengths)
            self.backbone.compile(dynamic=False, options={"triton.cudagraphs": False})

    def initialize_weights(self):
        """Apply the GPT-2 init.

        N(0, 0.02) everywhere, residual projections scaled by 1/sqrt(2 n_blocks). Padded embedding
        rows are zero (masked -inf, never looked up, zero gradient).
        """
        n_blocks = len(self.backbone.blocks)

        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        for block in self.backbone.blocks:
            nn.init.normal_(block.attn_out.weight, std=0.02 / math.sqrt(2 * n_blocks))
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / math.sqrt(2 * n_blocks))
        nn.init.normal_(self.backbone.wte.weight, std=0.02)
        with torch.no_grad():
            self.backbone.wte.weight[self.real_vocab_size :] = 0.0

    # Optimizer routing
    def muon_spec(self):
        """Return the (Muon name regex, AdamW 2D names, expected Muon count) routing spec.

        Consumed by `optimizers.split_muon_params`: Muon = the per-block hidden matrices; AdamW =
        the tied embedding/head, norm gains.
        """
        pattern = r'^backbone\.blocks\.\d+\.(attn_qkv|attn_out|mlp\.fc1|mlp\.fc2)\.weight$'
        return pattern, {'backbone.wte.weight'}, 4 * len(self.backbone.blocks)

    @property
    def device(self):
        return self.backbone.wte.weight.device

    # Training
    def forward(self, input_ids, attention_mask):
        feats = self.backbone(input_ids)  # (B, L, dim) bf16
        dim = feats.shape[-1]
        hidden = feats[:, :-1].reshape(-1, dim)
        targets = input_ids[:, 1:].reshape(-1)
        row_weight = attention_mask[:, 1:].reshape(-1).float()
        loss_sum = chunked_linear_cross_entropy(
            hidden, self.backbone.wte.weight, None, targets, row_weight, col_mask=self.logit_mask
        )
        return {'loss': loss_sum / row_weight.sum()}

    # Sampling
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None):
        """Sample ancestrally from BOS with TF32 off; returns (B, num_steps + 1) incl. the BOS."""
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._generate(num_samples, num_steps)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

    def _generate(self, num_samples, num_steps):
        cfg = self.relf_config.sampling
        if num_steps is None:
            num_steps = self.num_tokens - 1
        if num_steps > self.num_tokens - 1:
            raise ValueError(f'num_steps {num_steps} exceeds the context of {self.num_tokens} tokens')
        temperature = float(cfg.get('temperature', 1.0))
        top_k = cfg.get('top_k', None)
        top_p = cfg.get('top_p', None)
        if temperature <= 0:
            raise ValueError('sampling.temperature must be > 0')
        was_training = self.training
        self.eval()
        try:
            B, device = num_samples, self.device
            bb = self.backbone
            head_dim = bb.dim // bb.n_heads
            caches = [
                (
                    torch.zeros(B, num_steps + 1, bb.n_heads, head_dim, dtype=torch.bfloat16, device=device),
                    torch.zeros(B, num_steps + 1, bb.n_heads, head_dim, dtype=torch.bfloat16, device=device),
                )
                for _ in bb.blocks
            ]
            tokens = torch.full((B, num_steps + 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)
            w = bb.wte.weight.float()
            for pos in range(num_steps):
                feats = bb.decode(tokens[:, pos : pos + 1], caches, pos)
                logits = F.linear(feats.float(), w)
                if self.logit_mask is not None:
                    logits = logits + self.logit_mask
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float('-inf'))
                if top_p is not None:
                    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
                    # Drop a token once the mass ranked strictly above it exceeds `top_p`, so the
                    # token that crosses the threshold is kept
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
                    drop = cum - torch.softmax(sorted_logits, dim=-1) > float(top_p)
                    sorted_logits = sorted_logits.masked_fill(drop, float('-inf'))
                    logits = torch.full_like(logits, float('-inf')).scatter(-1, order, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                tokens[:, pos + 1] = torch.multinomial(probs, 1)[:, 0]
            return tokens
        finally:
            self.train(was_training)

    # Checkpoints
    def load_checkpoint_state_dict(self, state_dict):
        state_dict = utils.filter_compile_state_dict(state_dict)
        return self.load_state_dict(state_dict, strict=True)
