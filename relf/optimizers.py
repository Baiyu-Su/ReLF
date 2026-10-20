"""Optimizer construction for ReLF training.

`build_optimizer` dispatches on `optim.name`: `adamw` (plain AdamW over all trainable parameters)
or `muon` (the Muon + AdamW split below); anything else raises.

The Muon split is by explicit parameter name, never the `ndim == 2` heuristic of the torch docs
example -- that heuristic would also route the output head and `input_proj` to Muon, exactly the
layers Keller Jordan's and Moonshot's practice keeps on AdamW:

- **Muon**: every 2D weight except the two below -- the per-block hidden matrices (`attn_qkv`,
  `attn_gate` when gated attention is enabled, `attn_out`, `mlp.fc1`, `mlp.fc2`), all adaLN
  modulation weights (blocks + output layer), the time-embedder MLP, and the guidance-scale
  embedder MLP (`sc_w_embed`, `algo.sc_guidance`).
- **AdamW**: the output head (zero-init, 50304 x 768, emits unbounded logits that Muon's
  spectral-norm-constraining update fights), `input_proj` (the de facto learnable input
  embedding; `token_embed` is a frozen buffer), every norm gain, and every bias.

Both groups share `optim.lr` and one schedule (`trainer.lr_scheduler`; the WSD 1-sqrt cooldown is
`build_wsd_scheduler` below): Muon runs with `adjust_lr_fn='match_rms_adamw'` (Moonshot's
0.2 * lr * sqrt(max(fan_in, fan_out)) update-RMS matching), so there is no separate Muon learning
rate to tune.

When `model.compile` is on (the single global compile switch), the torch Newton-Schulz kernel is
torch.compile'd in place (Kimi/Moonlight-style) -- see `_compile_newton_schulz`.
"""

import math
import re

import torch

# Every 2D weight goes to Muon except the explicit AdamW matrices below (`input_proj` and the LM
# head); norm gains and biases are 1D and stay AdamW
_MUON_PATTERN = re.compile(
    r'^backbone\.('
    r'blocks\.\d+\.(attn_qkv|attn_gate|attn_out|mlp\.fc1|mlp\.fc2)\.weight'
    r'|(blocks\.\d+|output_layer)\.adaLN_modulation\.\d+\.weight'
    r'|(time_embed|sc_w_embed)\.mlp\.\d+\.weight'
    r')$'
)
_ADAMW_MATRICES = {
    'backbone.input_proj.weight',
    'backbone.output_layer.linear.weight',
}


def _muon_spec(model):
    """Return the (pattern, AdamW 2D names, expected Muon count) routing spec for the ReLF/DiT."""
    n_blocks = model.relf_config.model.n_blocks
    # 4 hidden matrices per block (+1 `attn_gate` under `model.gated_attention`), adaLN in every
    # block + the output layer, the two time-embedder MLP weights, and the two guidance-scale
    # embedder MLP weights under `algo.sc_guidance`
    per_block = 4 + int(bool(model.relf_config.model.get('gated_attention', False)))
    expected = per_block * n_blocks + (n_blocks + 1) + 2
    expected += 2 * int(bool(model.relf_config.algo.get('sc_guidance', False)))
    return _MUON_PATTERN, _ADAMW_MATRICES, expected


_ns_compiled = False


def _compile_newton_schulz():
    """Wrap the Newton-Schulz kernel inside `torch.optim.Muon` in `torch.compile`.

    torch 2.12's `_single_tensor_muon` runs NS eager: ~1,200 tiny kernels per step over the Muon
    matrices, measured at ~48 ms/step (+13%) on GH200. Compiling the module-level `_zeropower_via_newtonschulz` -- the same thing Kimi's
    Moonlight does to its NS function -- fuses each distinct matrix shape into one graph. The patch
    is safe in our stack: the `_disable_dynamo_if_unsupported` wrapper around `muon()` only
    disables compilation when the training loop itself is being traced
    (`torch.compiler.is_compiling()`), which never happens here -- `model.compile` compiles the DiT
    forward only. Idempotent, so repeated `build_optimizer` calls never double-wrap.
    """
    global _ns_compiled
    if _ns_compiled:
        return
    import torch.optim._muon as _torch_muon

    _torch_muon._zeropower_via_newtonschulz = torch.compile(
        _torch_muon._zeropower_via_newtonschulz, dynamic=False
    )
    _ns_compiled = True


def split_muon_params(model):
    """Partition trainable parameters into (muon, adamw) name/param lists.

    The routing spec comes from `model.muon_spec()` when the model defines one (`ar.GPT`: the
    per-block hidden matrices to Muon, the tied embedding/head to AdamW), else the ReLF/DiT spec
    above. Raises if either side is not exactly the expected set, so a new module can never
    silently land in the wrong group.
    """
    if hasattr(model, 'muon_spec'):
        pattern, adamw_matrices, expected = model.muon_spec()
        pattern = re.compile(pattern)
    else:
        pattern, adamw_matrices, expected = _muon_spec(model)
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pattern.match(name):
            if p.ndim != 2:
                raise ValueError(f'Muon-routed parameter {name} has ndim {p.ndim}, expected 2')
            muon.append((name, p))
        else:
            if p.ndim == 2 and name not in adamw_matrices:
                raise ValueError(
                    f'Unrouted 2D parameter {name}: extend the Muon pattern or the AdamW matrices'
                )
            adamw.append((name, p))
    if len(muon) != expected:
        raise ValueError(
            f'Muon split matched {len(muon)} matrices, expected {expected} '
            f'({[n for n, _ in muon]})'
        )
    return muon, adamw


class SplitOptimizer(torch.optim.Optimizer):
    """Two real optimizers behind one `torch.optim.Optimizer` facade.

    `torch.optim.Muon` has no per-group AdamW fallback, so Muon and AdamW are built separately and
    this wrapper makes them look like a single optimizer to `transformers.Trainer` / accelerate /
    the LR scheduler: `param_groups` is a LIVE view of the sub-optimizers' group dicts (one LambdaLR
    mutates both groups' lr in place), `step`/`zero_grad` delegate, and the state dict nests one
    entry per sub-optimizer so `optimizer.pt` save/resume round-trips. Resuming with a different
    `optim.name` fails loudly on the state-dict shape, as it should.

    `param_groups` MUST be a property, never a stored list: `Optimizer.load_state_dict` rebuilds
    each sub-optimizer's group dicts as new objects, and accelerate's `AcceleratedOptimizer`
    performs a `state_dict`/`load_state_dict` round-trip at `prepare()` time for device placement.
    A snapshot list taken at construction goes stale right there, and the LR scheduler then drives
    orphaned dicts while the real optimizers stay frozen at the snapshot lr -- 0.0 under warmup,
    i.e. no training at all.
    """

    def __init__(self, optimizers):
        # No `super().__init__` on purpose: it would build its own `param_groups`/`state`; ours
        # must mirror the sub-optimizers'
        self.optimizers = list(optimizers)
        self.defaults = {}

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    @property
    def state(self):
        merged = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'sub_optimizers': [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        subs = state_dict['sub_optimizers']
        if len(subs) != len(self.optimizers):
            raise ValueError(f'Expected {len(self.optimizers)} sub-states, got {len(subs)}')
        for opt, sd in zip(self.optimizers, subs):
            opt.load_state_dict(sd)

    def add_param_group(self, param_group):
        raise NotImplementedError('SplitOptimizer param groups are fixed at construction')

    def __repr__(self):
        return 'SplitOptimizer(\n' + '\n'.join(repr(o) for o in self.optimizers) + '\n)'


def build_optimizer(model, config):
    """Build the optimizer named by `optim.name` (`adamw` or `muon`); the single entry point."""
    name = config.optim.name
    if name == 'adamw':
        return _build_adamw(model, config)
    if name == 'muon':
        return _build_muon_split(model, config)
    raise ValueError(f"optim.name must be 'adamw' or 'muon', got {name!r}")


def _build_adamw(model, config):
    opt_cfg = config.optim
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    opt = torch.optim.AdamW(
        _adamw_groups(named, opt_cfg),
        lr=opt_cfg.lr,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    print(
        f'AdamW: {len(params)} tensors / {sum(p.numel() for p in params):,} params '
        f'(wd={opt_cfg.weight_decay}, lr={opt_cfg.lr})'
    )
    return opt


def _no_decay(p, opt_cfg):
    """Return True for 1D tensors (norm gains, biases) under `optim.decay_only_matrices`."""
    return bool(opt_cfg.get('decay_only_matrices', False)) and p.ndim < 2


def _adamw_groups(named, opt_cfg):
    """Build the AdamW param groups.

    One group by default, or a decayed + an undecayed group under `optim.decay_only_matrices` (the
    GPT-2 / nanoGPT convention: decay every matrix incl. the tied embedding, never norm gains or
    biases). Two groups change `optimizer.pt`'s shape, so the flag is whole-run only.
    """
    if not opt_cfg.get('decay_only_matrices', False):
        return [p for _, p in named]
    decay = [p for _, p in named if not _no_decay(p, opt_cfg)]
    no_decay = [p for _, p in named if _no_decay(p, opt_cfg)]
    return [
        {'params': decay, 'weight_decay': opt_cfg.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def _build_muon_split(model, config):
    opt_cfg = config.optim
    # The global compile switch also covers the optimizer: no separate knob
    if config.model.get('compile', False):
        _compile_newton_schulz()
    muon_named, adamw_named = split_muon_params(model)
    muon = torch.optim.Muon(
        [p for _, p in muon_named],
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.muon_weight_decay,
        momentum=opt_cfg.muon_momentum,
        nesterov=bool(opt_cfg.muon_nesterov),
        ns_steps=int(opt_cfg.muon_ns_steps),
        adjust_lr_fn='match_rms_adamw',
    )
    adamw = torch.optim.AdamW(
        _adamw_groups(adamw_named, opt_cfg),
        lr=opt_cfg.lr,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    n_muon = sum(p.numel() for _, p in muon_named)
    n_adamw = sum(p.numel() for _, p in adamw_named)
    n_nodecay = sum(p.numel() for _, p in adamw_named if _no_decay(p, opt_cfg))
    print(
        f'SplitOptimizer: Muon {len(muon_named)} matrices / {n_muon:,} params '
        f'(wd={opt_cfg.muon_weight_decay}, match_rms_adamw), '
        f'AdamW {len(adamw_named)} tensors / {n_adamw:,} params '
        f'(wd={opt_cfg.weight_decay}, {n_nodecay:,} of them undecayed); shared lr={opt_cfg.lr}; '
        f'NS {"compiled" if _ns_compiled else "eager"}'
    )
    return SplitOptimizer([muon, adamw])


def build_wsd_scheduler(optimizer, config, num_warmup_steps, num_training_steps):
    """Build a warmup-stable-decay LambdaLR with a 1-sqrt cooldown.

    The cooldown follows Hagele et al. 2024 (arXiv:2405.18392). Multiplier of the peak lr, with
    r = `optim.final_lr` / `optim.lr`: linear warmup, then 1.0, then
    `r + (1 - r) * (1 - sqrt(progress))` over the last `trainer.decay_steps`. It reaches r at
    `num_training_steps`, one step after the last step taken.
    """
    if config.trainer.get('decay_steps') is None:
        raise ValueError('warmup_stable_decay requires trainer.decay_steps')
    if config.optim.get('final_lr') is None:
        raise ValueError('warmup_stable_decay requires optim.final_lr')
    num_decay_steps = int(config.trainer.decay_steps)
    peak_lr = float(config.optim.lr)
    final_lr = float(config.optim.final_lr)
    if not 0 < num_decay_steps <= num_training_steps - num_warmup_steps:
        raise ValueError(
            f'trainer.decay_steps must be in (0, max_steps - warmup_steps]; got '
            f'{num_decay_steps} with max_steps={num_training_steps}, '
            f'warmup_steps={num_warmup_steps}'
        )
    if not 0 <= final_lr <= peak_lr:
        raise ValueError(f'optim.final_lr must be in [0, optim.lr={peak_lr}]; got {final_lr}')
    min_lr_ratio = final_lr / peak_lr
    decay_start = num_training_steps - num_decay_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < decay_start:
            return 1.0
        progress = min(1.0, (current_step - decay_start) / num_decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
