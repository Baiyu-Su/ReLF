"""Continuous flow language model (`ReLF`): asset, t density, corruption, and the training objective.

The fused head+CE lives in fused_ce.py, the inference samplers in samplers.py (`SamplerMixin`).
"""

import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from relf import utils
from relf.fused_ce import (
    CE_CHUNK_ROWS,
    SINGULAR_TRAIN_MIN_ONE_MINUS_T,
    _add_singular_logits,
    _exact_singular_ce,
    _slice_singular,
    chunked_linear_cross_entropy,
)
from relf.model import DIT
from relf.samplers import SamplerMixin


class ReLF(SamplerMixin, nn.Module):
    """Single-time continuous flow language model.

    Plain `nn.Module` trained with `transformers.Trainer`: `forward(input_ids, attention_mask)`
    returns `{'loss'}`, the token-averaged cross-entropy of the batch. The flow model has no
    likelihood interpretation, so there is no NLL/perplexity of the data.
    """

    def __init__(self, config, tokenizer, vocab_size=None):
        super().__init__()
        # Not `config`: `transformers.Trainer` treats a `model.config` as a `PretrainedConfig`
        self.relf_config = config
        self.tokenizer = tokenizer
        # Head padded from the real vocab (50258) to `model.vocab_padded_size` for the GEMMs; padded
        # slots get zero embedding rows and a -inf `logit_mask` (zero gradient)
        self.real_vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.vocab_size = int(config.model.vocab_padded_size)
        self.backbone = DIT(config, vocab_size=self.vocab_size)
        # Frozen flow-space asset (`embedding`, `pi`, `schedule`); the embedding is a buffer saved
        # into checkpoints, never trained
        self.flow_dim = config.model.flow_dim
        blob = torch.load(config.model.token_embed_path, map_location='cpu')
        token_embed = blob['embedding'].float()
        assert token_embed.shape == (self.real_vocab_size, self.flow_dim), (
            token_embed.shape,
            self.real_vocab_size,
            self.flow_dim,
        )
        self.register_buffer('token_embed', F.pad(token_embed, (0, 0, 0, self.vocab_size - self.real_vocab_size)))
        pi = F.pad(blob['pi'].float(), (0, self.vocab_size - blob['pi'].shape[0]))
        # Additive logit offset wherever logits define a distribution: -inf on the padded slots, plus
        # log pi under `algo.singular_unigram_reference` (whole-run parameterization change)
        self.singular_logits = bool(config.algo.singular_logits)
        logit_mask = torch.zeros(self.vocab_size)
        logit_mask[self.real_vocab_size :] = float('-inf')
        if config.algo.singular_unigram_reference:
            if not self.singular_logits:
                raise ValueError('singular_unigram_reference requires algo.singular_logits')
            logit_mask = logit_mask + pi.clamp_min(1e-8).log()
        self.register_buffer('logit_mask', logit_mask, persistent=False)
        # pi-weighted mean of x1: the moment-matched start of grids whose first node is t_0 > 0
        self.register_buffer('prior_mean', (pi / pi.sum()) @ self.token_embed, persistent=False)
        # u -> alpha arc-length table; fp64 so that 1 - t = 1 / (1 + alpha) has no cancellation
        self.register_buffer('sched_alpha_of_u', blob['schedule']['alpha_of_u'].double(), persistent=False)
        self.sched_t_end = float(blob['schedule']['t_end'])
        if self.singular_logits:
            # (flow_dim + 1, V) = [E; ||E||^2]^T, paired with `_singular_inputs` in one matmul
            e_sq = (self.token_embed * self.token_embed).sum(dim=1, keepdim=True)
            self.register_buffer(
                'singular_embed',
                torch.cat([self.token_embed, e_sq], dim=1).t().contiguous(),
                persistent=False,
            )
        self.num_tokens = config.model.length
        self.antithetic_sampling = config.training.antithetic_sampling
        # Training t density only, never a CE weight; `t_max: 't_end'` reads the asset schedule
        self.t_sampling = utils.config_choice(config.training, 't_sampling', ('uniform', 'logit_normal', 'arc_length'))
        self.t_uniform_range = None
        t_support_max = None  # Upper end of the training-t support (None = unbounded)
        if self.t_sampling == 'logit_normal':
            self.t_logit_mean = float(config.training.t_logit_mean)
            self.t_logit_std = float(config.training.t_logit_std)
        elif self.t_sampling == 'arc_length':
            t_support_max = self.sched_t_end
        else:
            t_min = float(config.training.t_min)
            t_max = config.training.t_max
            t_max = self.sched_t_end if t_max == 't_end' else float(t_max)
            if not 0.0 <= t_min < t_max <= 1.0:
                raise ValueError(f'need 0 <= t_min < t_max <= 1, got ({t_min}, {t_max})')
            if (t_min, t_max) != (0.0, 1.0):
                self.t_uniform_range = (t_min, t_max)
            t_support_max = t_max
        # Densities reaching 1 - t < `SINGULAR_TRAIN_MIN_ONE_MINUS_T` screen each sequence in `forward`
        self._ce_exact_screen = self.singular_logits and (
            t_support_max is None or 1.0 - float(t_support_max) < SINGULAR_TRAIN_MIN_ONE_MINUS_T
        )
        self._init_sampler(config)
        # Self-conditioning and SC guidance;
        # parameterization changes -- whole-run only
        self.self_cond = bool(config.algo.self_cond)
        self.self_cond_p = float(config.algo.self_cond_p)
        self.sc_guidance = bool(config.algo.sc_guidance)
        self.omega_min = float(config.algo.omega_min)
        self.omega_max = float(config.algo.omega_max)
        self.sc_guidance_scale = float(config.sampling.sc_guidance_scale)
        self._sc_w_cache = {}
        if self.sc_guidance and not self.self_cond:
            raise ValueError('algo.sc_guidance requires algo.self_cond')
        if self.sc_guidance_scale != 1.0 and not self.sc_guidance:
            raise ValueError('sampling.sc_guidance_scale != 1 needs a model trained with algo.sc_guidance')
        # Reflow runs (reflow.py) train on stored (x0, tokens) couplings and must receive the x0 column
        reflow_cfg = config.get('reflow', None)
        self.require_x0 = bool(reflow_cfg is not None and reflow_cfg.get('enabled', False))
        self.global_batch_size = config.loader.global_batch_size
        self.grad_accum_steps = config.trainer.accumulate_grad_batches
        # Dedicated generator for `_sample_t`, seeded with the bare config seed on every rank: the
        # global t stratification requires all ranks to draw the identical vector, while the global
        # (per-rank-offset) RNG serves the corruption noise, dropout, and sampling. Created lazily
        # on the model's device; its state is checkpointed as t_rng.pt (see trainer.py).
        self.seed = config.seed
        self._t_generator = None
        # Stratified t vector for the current gradient-accumulation window, drawn once at
        # accum_step 0 and sliced by every rank/micro-step
        self._t_window = None
        # Micro-batch index within the current gradient-accumulation window, advanced by training
        # forwards and reset each epoch by the trainer
        self._accum_micro_step = 0

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def reset_accumulation_counter(self):
        self._accum_micro_step = 0

    @staticmethod
    def _world_info():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, 0

    def _t_rng(self):
        """Return the shared-stream generator behind `_sample_t`, created lazily on the device.

        Seeded with the bare config seed, never rank-offset, and consumed in identical shapes on
        every rank, so all ranks stay in lockstep and slice complementary chunks of the same
        stratified vector.
        """
        if self._t_generator is None or self._t_generator.device != self.device:
            self._t_generator = torch.Generator(device=self.device)
            self._t_generator.manual_seed(self.seed)
        return self._t_generator

    def t_rng_state(self):
        return self._t_rng().get_state()

    def set_t_rng_state(self, state):
        self._t_rng().set_state(state)

    def _stratified_t(self, n):
        """Draw `n` (antithetically stratified) t from the configured density via `_t_rng`."""
        g = self._t_rng()
        u = torch.rand(n, device=self.device, generator=g)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            u = (u / n + offset) % 1
            perm = torch.randperm(n, device=self.device, generator=g)
            u = u[perm]
        if self.t_sampling == 'logit_normal':
            # Inverse-CDF map of the stratified uniforms: the strata become equal-mass strata of
            # p(t), preserving the antithetic variance reduction; generator consumption is identical
            # to 'uniform', so the shared t stream stays rank-synchronized either way
            z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            u = torch.sigmoid(self.t_logit_mean + self.t_logit_std * z)
        elif self.t_sampling == 'arc_length':
            # Same inverse-CDF slot, same generator consumption: the monotone u -> t table turns the
            # strata into equal-mass strata of the schedule density p_train(t) = du/dt.
            u = self._lut_t(u)
        elif self.t_uniform_range is not None:
            # Range-restricted uniform: an affine map in the same slot, so the strata become
            # equal-mass strata of Uniform(t_min, t_max) and the generator consumption is unchanged
            t_min, t_max = self.t_uniform_range
            u = t_min + (t_max - t_min) * u
        return u

    def _lut_alpha(self, u):
        """Lerp alpha (fp64) from the asset's uniform-u alpha table, u in [0, 1].

        Working in alpha = t/(1-t) keeps 1 - t = 1/(1+alpha) exact in fp64 (the sampler's
        contraction factors need it).
        """
        lut = self.sched_alpha_of_u
        pos = u.double() * (lut.numel() - 1)
        lo = pos.floor().long().clamp_(0, lut.numel() - 2)
        return torch.lerp(lut[lo], lut[lo + 1], pos - lo)

    def _lut_t(self, u):
        """Map u -> t = alpha/(1+alpha) monotonically, fp32 out; t_max = t_end."""
        alpha = self._lut_alpha(u)
        return (alpha / (1.0 + alpha)).float()

    def _sample_t(self, n, accum_step):
        """Sample t from the configured density on [0, 1); t = 0 is pure noise, t = 1 is data.

        When `accum_step` is given, the antithetic stratification spans the full global batch: a
        single stratified vector of size `global_batch_size` is drawn from the shared-stream
        `_t_rng` generator once per accumulation window (at accum_step 0, identically on every rank)
        and this rank/micro-step slices its chunk. With `accum_step=None` (eval), the stratification
        covers only the local batch.
        """
        if accum_step is None:
            return self._stratified_t(n)
        if accum_step == 0 or self._t_window is None:
            self._t_window = self._stratified_t(self.global_batch_size)
        world_size, rank = self._world_info()
        t = self._t_window.chunk(world_size)[rank]
        t = t.chunk(self.grad_accum_steps)[accum_step]
        return t[:n]

    def corrupt_continuous(self, x1, t, x0=None):
        """Interpolate xt = t * embed(x1) + (1 - t) * gaussian noise.

        The interpolation lives in the fixed flow space (flow_dim), not in vocabulary space:
        embed(x1) is a lookup into `self.token_embed`.

        `x0` (optional, (B, L, flow_dim), any float dtype): a SUPPLIED source point instead of a
        fresh N(0, I) draw -- the reflow coupling hook (reflow.py): stored (x0, tokens) pairs of a
        teacher's deterministic map pair each sequence with its own x0. The interpolant itself is
        unchanged; only the (x0, x1) coupling is. None keeps the independent coupling and the
        historical RNG consumption byte for byte.
        """
        t = t.view(-1, 1, 1)
        x1_embed = self.token_embed[x1]
        if x0 is not None:
            if tuple(x0.shape) != tuple(x1_embed.shape):
                raise ValueError(f'x0 shape {tuple(x0.shape)} != {tuple(x1_embed.shape)}')
            return t * x1_embed + (1 - t) * x0.to(device=x1_embed.device, dtype=torch.float32)
        return t * x1_embed + (1 - t) * torch.randn_like(x1_embed)

    def _embed_matrix(self):
        """Return the fp32 (V, flow_dim) embedding for the softmax expectation."""
        return self.token_embed.float()

    def _softmax_x1(self, chunk, token_embed):
        """Return the fp32 softmax expectation of x1 for a chunk of (masked) logits: probs @ E."""
        return chunk.softmax(dim=-1, dtype=torch.float32) @ token_embed

    def _singular_desc(self, xt, t, eps=1e-5):
        """Build the descriptor `(xs_aug, singular_embed)` for `_add_singular_logits`."""
        return (self._singular_inputs(xt, t, eps).reshape(-1, self.flow_dim + 1), self.singular_embed)

    def _singular_inputs(self, xt, t, eps=1e-5):
        """Build the augmented state for the singular logits, shape (B, L, flow_dim + 1) fp32.

        The singular logits s[w] = -||xt - t * E[w]||^2 / (2 (1 - t)^2), dropping the w-independent
        ||xt||^2 term (a per-position constant that cancels in softmax/CE/argmax), expand to
        a * <xt, E[w]> + b * ||E[w]||^2 with a = t / (1 - t)^2 and b = -t^2 / (2 (1 - t)^2). This
        returns [a * xt, b] so that `inputs @ singular_embed` gives s. Everything stays fp32: the
        magnitudes diverge as t -> 1 and bf16 spacing there erases logit gaps.
        """
        t = t.float().clamp(max=1.0 - eps).view(-1, 1, 1)
        inv = (1.0 - t) ** -2
        xs = (t * inv) * xt.float()
        b = (-0.5 * t * t * inv).expand(xs.shape[0], xs.shape[1], 1)
        return torch.cat([xs, b], dim=-1)

    def _sc_w_const(self, B, device, value):
        """Return a cached constant (B,) fp32 guidance-scale vector for sampling."""
        key = (B, str(device), float(value))
        w = self._sc_w_cache.get(key)
        if w is None:
            w = torch.full((B,), float(value), device=device, dtype=torch.float32)
            self._sc_w_cache[key] = w
        return w

    def _backbone_forward(self, xt, t, x_sc=None, return_features=False, sc_w=None):
        """Call the backbone, handling the self-conditioning input.

        Runs outside any surrounding autocast so the flow/time embeddings stay in fp32; the DiT
        re-enters bf16 internally.

        `sc_w` (`algo.sc_guidance` only): per-example guidance scale, (B,) fp32. None fills the sampling
        constant `sampling.sc_guidance_scale` -- the single point where every sampler family gets the
        scale, at every forward (step 0 cold and the final argmax call included, as ELF).
        """
        if self.sc_guidance:
            if sc_w is None:
                sc_w = self._sc_w_const(xt.shape[0], xt.device, self.sc_guidance_scale)
        elif sc_w is not None:
            raise ValueError('sc_w passed to a model without algo.sc_guidance')
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.backbone(xt, t, x_sc=x_sc, return_features=return_features, sc_w=sc_w)

    @torch.no_grad()
    def _expectation_from_features(self, features, xt, t):
        """Compute the row-chunked head + fp32 softmax expectation from pre-head features.

        The training-time twin of the sampler's x1_hat (same singular addmm, same `logit_mask`, same
        fp32 softmax @ `token_embed`), never materializing the (B*L, V) logits. TF32 is forced off
        for the fp32 GEMMs so the training conditioning signal matches the sampler (TF32 is off
        process-wide, main.py).

        Autocast is disabled for the body: during training this runs inside accelerate's bf16
        autocast, which would silently downcast the final fp32 probs @ `token_embed` GEMM to bf16
        (the head `F.linear` stays a bf16 GEMM either way -- inputs are bf16).
        """
        B, L = features.shape[0], features.shape[1]
        head = self.backbone.output_layer.linear
        w16 = head.weight.detach().to(features.dtype)
        b16 = head.bias.detach().to(features.dtype) if head.bias is not None else None
        flat = features.reshape(-1, features.shape[-1])
        n_rows = flat.shape[0]
        token_embed = self._embed_matrix()
        sing = self._singular_desc(xt, t) if self.singular_logits else None
        x1 = torch.empty((n_rows, self.flow_dim), device=features.device, dtype=torch.float32)
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            for s in range(0, n_rows, CE_CHUNK_ROWS):
                e = min(s + CE_CHUNK_ROWS, n_rows)
                chunk = F.linear(flat[s:e], w16, b16).float()
                if sing is not None:
                    _add_singular_logits(chunk, sing, s, e)
                chunk.add_(self.logit_mask)
                x1[s:e] = self._softmax_x1(chunk, token_embed)
        return x1.view(B, L, self.flow_dim)

    def forward(self, input_ids, attention_mask, x0=None):
        """Return `{'loss'}`, the token-averaged CE of the batch.

        `x0` is an optional dataset column (reflow coupling datasets, dataloader.py): the
        per-sequence source point of a stored coupling. It is named in the signature on purpose --
        `transformers.Trainer` keeps only dataset columns that appear here. No `**kwargs` (the
        Trainer would otherwise stop normalizing the loss by grad-accum).
        """
        x1 = input_ids
        B = x1.shape[0]
        if self.training:
            accum_step = self._accum_micro_step % self.grad_accum_steps
            self._accum_micro_step += 1
        else:
            accum_step = None
        t = self._sample_t(B, accum_step)
        if self.training and self.require_x0 and x0 is None:
            # A reflow run whose batches arrive without x0 would silently train under the
            # independent coupling (Trainer column filtering, collator, or a dataset slip). Fail
            # loudly instead.
            raise RuntimeError('reflow.enabled but forward() received x0=None')
        x_t = self.corrupt_continuous(x1, t, x0=x0)
        x_sc = None
        sc_w = None
        hidden_shift = None
        if self.sc_guidance:
            # Guidance scale, log-uniform over 1 + w on [1 + min, 1 + max] (ELF `sample_cfg_scale`),
            # from the rank-offset global CUDA RNG like the bootstrap mask below -- never from the
            # shared t stream
            u = torch.rand(B, device=x_t.device)
            a, b = 1.0 + self.omega_min, 1.0 + self.omega_max
            sc_w = a * torch.exp(u * math.log(b / a)) - 1.0
        if self.self_cond:
            # Bootstrap pass (Analog Bits Alg. 1): one no-grad cold-mode forward at the SAME
            # (x_t, t) supplies the conditioning signal; a per-example Bernoulli(self_cond_p) mask
            # keeps the cold mode trained. The mask and the pass's dropout draw from the rank-offset
            # global CUDA RNG -- never from the shared t stream (`_t_rng` consumption must stay
            # identical on every rank). Calls `_backbone_forward` directly: a second `ReLF.forward`
            # would advance `_accum_micro_step` and desync the t-window slicing. Training thus
            # exercises exactly the two modes the sampler uses (step 0 fully cold, later steps fully
            # warm).
            #
            # `sc_guidance` adds a second no-grad pass in the warm mode (conditioned on the UNMASKED cold
            # estimate, as ELF) and forms the stop-grad feature shift (1 - 1/w) keep
            # (f_warm - f_cold) that the CE subtracts. Both no-grad
            # passes then run with dropout off (ELF deterministic=True): CE is nonlinear in the
            # shift, so dropout noise in it would bias the distillation fixed point. The
            # eval()/train() toggle is gated on `sc_guidance` so plain self-cond runs keep their exact
            # bootstrap; under torch.compile it costs one extra cached graph, not a per-call
            # recompile.
            was_training = self.backbone.training
            if self.sc_guidance and was_training:
                self.backbone.eval()
            try:
                with torch.no_grad():
                    feats0 = self._backbone_forward(x_t, t, x_sc=None, return_features=True, sc_w=sc_w)
                    keep = torch.rand(B, 1, 1, device=x_t.device) < self.self_cond_p
                    x1_boot = self._expectation_from_features(feats0, x_t, t)
                    x_sc = torch.where(keep, x1_boot, torch.zeros_like(x1_boot))
                    if self.sc_guidance:
                        feats_c = self._backbone_forward(x_t, t, x_sc=x1_boot, return_features=True, sc_w=sc_w)
                        coef = (1.0 - 1.0 / sc_w).view(B, 1, 1) * keep
                        hidden_shift = (coef * (feats_c.float() - feats0.float())).reshape(
                            -1, feats_c.shape[-1]
                        )
                        del feats_c
                    del feats0
            finally:
                if self.sc_guidance and was_training:
                    self.backbone.train()
        # Chunked fused head+CE: the (B*L, V) logits never exist at once
        features = self._backbone_forward(x_t, t, x_sc=x_sc, return_features=True, sc_w=sc_w)
        head = self.backbone.output_layer.linear
        singular = self._singular_desc(x_t, t) if self.singular_logits else None
        hidden = features.reshape(-1, features.shape[-1])
        targets = x1.reshape(-1)
        row_weight = attention_mask.reshape(-1).float()
        # Sequences drawn too close to t = 1 for the fused kernel (`SINGULAR_TRAIN_MIN_ONE_MINUS_T`)
        # take the exact fp32 CE; the rest keep the fused path. t is per sequence, so the split is
        # by whole L-row blocks. Only densities that can reach the region pay the per-step host
        # sync.
        exact_rows = None
        if self._ce_exact_screen:
            exact_seq = (1.0 - t.float()) < SINGULAR_TRAIN_MIN_ONE_MINUS_T
            if bool(exact_seq.any()):
                exact_rows = exact_seq.view(-1, 1).expand(B, x1.shape[1]).reshape(-1)
        if exact_rows is None:
            loss_sum = chunked_linear_cross_entropy(
                hidden,
                head.weight,
                head.bias,
                targets,
                row_weight,
                singular=singular,
                col_mask=self.logit_mask,
                hidden_shift=hidden_shift,
            )
        else:
            keep = ~exact_rows
            loss_sum = _exact_singular_ce(
                hidden[exact_rows],
                head.weight,
                head.bias,
                targets[exact_rows],
                row_weight[exact_rows],
                _slice_singular(singular, exact_rows),
                self.logit_mask,
                hidden_shift=hidden_shift[exact_rows] if hidden_shift is not None else None,
            )
            if bool(keep.any()):
                loss_sum = loss_sum + chunked_linear_cross_entropy(
                    hidden[keep],
                    head.weight,
                    head.bias,
                    targets[keep],
                    row_weight[keep],
                    singular=_slice_singular(singular, keep),
                    col_mask=self.logit_mask,
                    hidden_shift=hidden_shift[keep] if hidden_shift is not None else None,
                )
        num_tokens = attention_mask.sum()
        return {'loss': loss_sum / num_tokens}

    def load_checkpoint_state_dict(self, state_dict):
        """Load model weights (strict), stripping torch.compile key prefixes.

        Strict on purpose: a plain self-cond checkpoint must not load into an `sc_guidance` model with
        `sc_w_embed` left at fresh init (and vice versa) -- parameterization changes are whole-run.
        """
        return self.load_state_dict(utils.filter_compile_state_dict(state_dict), strict=True)
