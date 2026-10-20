"""DiT backbone of the flow model (time / guidance embedders, adaLN blocks) and parameter EMA."""

import math

import einops
from flash_attn.cute import flash_attn_func
import torch
import torch.nn as nn
from quack.mlp import MLP as FusedMLP
from quack.rmsnorm import rmsnorm as quack_rmsnorm
from quack.rotary import apply_rotary_emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            self.cos_cached = freqs.cos().to(x.device)
            self.sin_cached = freqs.sin().to(x.device)

        return self.cos_cached, self.sin_cached


# Layers
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return quack_rmsnorm(x, self.weight, eps=self.eps)


# Timestep embedding
class TimestepEmbedder(nn.Module):
    """Scalar -> vector embedding: sinusoidal features through a two-layer MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        `t` is a 1-D tensor of N values, one per batch element (may be fractional); `max_period`
        sets the minimum frequency. Returns an (N, dim) fp32 tensor.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # The sinusoidal embedding is always fp32; match the `mlp` weights so a bf16-cast model does
        # not hit a dtype mismatch (no-op for fp32)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


# Core model
class DDiTBlock(nn.Module):
    def __init__(
        self, dim, n_heads, cond_dim=None, mlp_ratio=4, dropout=0.1, gated_attention=False
    ):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = RMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        # Gated attention (Qiu et al. 2025, arXiv:2505.06708; Qwen3-Next):
        # the paper's best cell only -- a query-dependent, ELEMENTWISE sigmoid gate on the SDPA
        # output, before `attn_out` ("G1", multiplicative, non-shared across heads). The gate reads
        # the same normalized+modulated input as q/k/v; the reference fuses it into `q_proj`, here
        # it is a separate Linear so q/k/v keep the exact baseline init and the gate is the only
        # delta. Headwise / additive / other positions from the paper's ablations are NOT offered.
        # `model.gated_attention` False (default) is standard attention.
        self.gated_attention = gated_attention
        self.attn_gate = nn.Linear(dim, dim, bias=False) if gated_attention else None
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(dim // n_heads)
        self.k_norm = RMSNorm(dim // n_heads)

        self.norm2 = RMSNorm(dim)
        # Fused SwiGLU MLP (quack GemmGated); biases must stay off for the fused training path.
        # Width rounds UP to a multiple of 64: quack gates the fused kernels on
        # `fc1_out % 16 == 0`, and the raw 2/3 rule violates that at e.g. dim=1600 (4266 -> silent
        # eager fallback in every block). dim=768 gives 2048 and is unchanged by the rounding.
        _mlp_hidden = -(-int(mlp_ratio * dim * 2 / 3) // 64) * 64
        self.mlp = FusedMLP(
            dim,
            hidden_features=_mlp_hidden,
            activation="swiglu",
            concat_layout=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True))

    def forward(self, x, rotary_cos_sin=None, c=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)

        x_skip = x
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x_mod)
        qkv = einops.rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        q, k, v = qkv.unbind(dim=2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = rotary_cos_sin
        q = apply_rotary_emb(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rotary_emb(k, cos.to(k.dtype), sin.to(k.dtype))

        x = flash_attn_func(q, k, v, causal=False)
        x = x[0] if isinstance(x, tuple) else x

        x = einops.rearrange(x, 'b s h d -> b s (h d)')
        if self.gated_attention:
            # Elementwise sigmoid gate on the SDPA output (pre-`attn_out`), computed from the
            # attention-layer input `x_mod` -- query-dependent, per channel
            x = x * torch.sigmoid(self.attn_gate(x_mod))

        x = x_skip + gate_msa.unsqueeze(1) * self.dropout(self.attn_out(x))
        x = x + gate_mlp.unsqueeze(1) * self.dropout(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )

    def features(self, x, c):
        """Return the modulated, normalized features right before the vocab projection."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return modulate(self.norm_final(x), shift, scale)

    def forward(self, x, c):
        return self.linear(self.features(x, c))


class DIT(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        # Self-conditioning (`algo.self_cond`): the input projection additionally consumes the
        # model's own previous x1_hat estimate, concatenated on the feature axis (Analog Bits,
        # arXiv 2208.04202). The x1_hat columns are zero-init (see `initialize_weights`), so at
        # init the function is exactly the unconditioned baseline for any conditioning value.
        self.self_cond = bool(config.algo.get('self_cond', False))
        # self-conditioning guidance (`algo.sc_guidance`; after ELF, arXiv:2605.10938): a
        # scalar guidance scale w conditions the network like the timestep, through a second
        # `TimestepEmbedder` whose output is added to the adaLN vector c. Its last layer is
        # zero-init (see `initialize_weights`), so at init the function is the plain self-conditioned
        # model for every w.
        self.sc_guidance = bool(config.algo.get('sc_guidance', False))
        # Learnable projection from the flow space (fixed token embedding,
        # `config.model.flow_dim`) into the model width
        in_dim = config.model.flow_dim * (2 if self.self_cond else 1)
        self.input_proj = nn.Linear(in_dim, dim, bias=True)
        self.time_embed = TimestepEmbedder(cond_dim)
        if self.sc_guidance:
            self.sc_w_embed = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    dropout=config.model.dropout,
                    gated_attention=bool(config.model.get('gated_attention', False)),
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim, out_channels=vocab_size, cond_dim=cond_dim
        )

        self.initialize_weights()

        # Single global compile switch (`config.model.compile`): compiles the whole DiT -- blocks,
        # embedders, final layer -- or nothing. Default mode: the win here is pointwise fusion of
        # the adaLN/residual glue; the heavy GEMMs are custom quack/FA4 kernels that Inductor
        # cannot tune anyway, so `max_autotune` is not worth its compile time.
        if config.model.compile:
            self.compile(dynamic=False, options={"triton.cudagraphs": False})

    def initialize_weights(self):
        # JiT/DiT-style base init: xavier-uniform every Linear (including the quack `FusedMLP`
        # fc1/fc2 and the attention projections), zero biases
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if self.sc_guidance:
            # Guidance-scale embedder: first layer as the time embedder, last layer zero so
            # `sc_w_embed(w) == 0` at init for every w (its input `SiLU(mlp[0](freq))` is nonzero,
            # so the zero layer still trains)
            nn.init.normal_(self.sc_w_embed.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.sc_w_embed.mlp[2].weight)

        # Zero-init the adaLN modulation layers so blocks start as identity
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-init the output layer, so the initial logits are zero
        nn.init.zeros_(self.output_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.output_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.output_layer.linear.weight)
        if self.output_layer.linear.bias is not None:
            nn.init.zeros_(self.output_layer.linear.bias)

        # Self-conditioning columns (must run AFTER the xavier sweep above, or `apply()` silently
        # overwrites the zeros): the x_t half is re-drawn at the unwidened fan-in so its init
        # statistics match a non-self-cond model; the x1_hat half is zeroed, so at init the
        # network is exactly the unconditioned baseline and opts into the signal by gradient
        if self.self_cond:
            flow_dim = self.config.model.flow_dim
            base = torch.empty(self.input_proj.weight.shape[0], flow_dim)
            nn.init.xavier_uniform_(base)
            with torch.no_grad():
                self.input_proj.weight[:, :flow_dim] = base
                self.input_proj.weight[:, flow_dim:] = 0.0

    def forward(self, x, t, x_sc=None, return_features=False, sc_w=None):
        """Return logits, or with `return_features=True` the pre-head features.

        The features feed the chunked fused linear+CE training loss, which applies
        `output_layer.linear` itself chunk by chunk.

        `sc_w` (sc_guidance only): the self-conditioning guidance scale, (B,) fp32, embedded and added
        to the adaLN vector. Required when `algo.sc_guidance` is on (`ReLF._backbone_forward` fills the
        sampling default).

        `x_sc` (self_cond only): the previous x1_hat estimate, (B, L, flow_dim); None means the
        cold mode (zeros -- sampler step 0 / the unconditioned half of training).
        """
        # The flow-space input and time embedding are computed in fp32; the residual stream is
        # bf16 from here through the head, so norms, modulation, residual adds, pre-head features,
        # and the vocab projection all run in bf16
        if self.self_cond:
            if x_sc is None:
                x_sc = torch.zeros_like(x)
            x = torch.cat([x, x_sc.to(x.dtype)], dim=-1)
        x = self.input_proj(x.to(self.input_proj.weight.dtype)).to(torch.bfloat16)
        c = self.time_embed(t)
        if self.sc_guidance:
            c = c + self.sc_w_embed(sc_w)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=c)
            x = self.output_layer.features(x, c=c)
            if not return_features:
                x = self.output_layer.linear(x)

        return x


# EMA
class ExponentialMovingAverage:
    """Exponential moving average of a set of parameters (only those with `requires_grad`)."""

    def __init__(self, parameters, decay):
        """Snapshot `parameters` (usually `model.parameters()`) as the initial shadow copy.

        `decay` must lie in [0, 1]; the effective decay is warmed up as
        min(decay, (1 + n) / (10 + n)) over the update count n.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def move_shadow_params_to_device(self, device):
        self.shadow_params = [i.to(device) for i in self.shadow_params]

    def update(self, parameters):
        """Fold `parameters` into the shadow copy.

        Call this after every parameter update (each `optimizer.step()`), with the same parameter
        set used to initialize this object.
        """
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            # Single multi-tensor kernel instead of a Python loop over ~200 params:
            # shadow = decay * shadow + (1 - decay) * param
            torch._foreach_lerp_(self.shadow_params, parameters, one_minus_decay)

    def copy_to(self, parameters):
        """Copy the moving averages into `parameters`."""
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def store(self, parameters):
        """Save a copy of `parameters` for a later `restore`."""
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """Restore the parameters saved by `store` into `parameters`.

        Lets the model be validated (or saved) with EMA weights without affecting the optimization:
        `store`, then `copy_to`, evaluate, then `restore`.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(
            decay=self.decay, num_updates=self.num_updates, shadow_params=self.shadow_params
        )

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']
