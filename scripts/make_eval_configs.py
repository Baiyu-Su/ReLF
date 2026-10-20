"""Generate all-literal sample_eval yamls for every checkpoint of a training run.

One yaml per (checkpoint, cell) so each is a bare `--config-name`
launch (no CLI overrides on real evals) and each process is short (one T per cell), so a
preempted launcher that skips completed cells loses at most one. The yamls are the training config with only `mode` / `run_name` /
`eval.checkpoint_path` / `sampling.*` / `eval_sampler.*` / `loader.eval_batch_size` changed, so the model section
(`flow_dim`, asset, gated attention, `self_cond`) always matches the checkpoint.

    python scripts/make_eval_configs.py --train-config relf_s --run-glob 'outputs/relf_s_2*'
    python scripts/make_eval_configs.py --train-config relf_s --run-glob 'outputs/relf_s_2*' \
        --steps 80000 --cells sde,tsweep

Cells (core matrix), one T per config by default: euler_T{16,32,64,128,256}
warm carry; sde = sde(noise 4.0) T=256; cold = euler T=256 cold carry (7 cells); heun_T<T> = Heun
(2 NFE/step). Extended cells: sde{2,3,6,8} = other noise scales at T=256; mg_a{alpha}_b{beta}_T{T}
= euler + MG; the frozen 64-NFE schedule and its `_scw<w>` self-conditioning guidance variants (token
grammar below). Every cell: EMA weights, 16 x 256 = 4096 samples, seed 1, gpt2-large scorer,
uniform grid unless a `_gknots` token says otherwise.
"""
import argparse
import glob
import os
import re
import sys

from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The core matrix as single-T cells (one process each), so a preemption loses at most one cell. 'tsweep' remains as a multi-T convenience.
CORE_CELLS = {
    'euler_T16': dict(steps=[16], solver='euler'),
    'euler_T32': dict(steps=[32], solver='euler'),
    'euler_T64': dict(steps=[64], solver='euler'),
    'euler_T128': dict(steps=[128], solver='euler'),
    'euler_T256': dict(steps=[256], solver='euler'),
    'sde': dict(steps=[256], solver='sde', sde_noise_scale=4.0),
    'cold': dict(steps=[256], solver='euler', carry_mode='cold'),
    'tsweep': dict(steps=[16, 32, 64, 128, 256], solver='euler'),
}
CORE_DEFAULT = 'euler_T16,euler_T32,euler_T64,euler_T128,euler_T256,sde,cold'


# The frozen 64-NFE schedule as a cell name, e.g.
#   sde32_T64_gknots0.13_0.24_0.7_0.6_a0.06_b0.56_nplate2_dt0.44_x24.5_esstd_mid
# = <base cell>[_gknots<u1>_<v1>_..._<uK>_<vK>_a<a>_b<b>][_np<profile>][_dt<t>][_x2<c>][_es<mode>]
#   [_scw<w>][_mid]:
#   _gknots<knots>_a<a>_b<b>  piecewise-linear knot map between the pinned endpoints t_0 = a,
#                           t_{N-1} = b (`sampling.grid` knots, `grid_knots`, `grid_a`, `grid_b`);
#                           needs `_mid`
#   _np<profile>[x<f>]      coupled, mean-preserving noise profile: both t-gates =
#                           `PROFILES[profile]` with `sde_gate_normalize` (mh state form); the
#                           optional x<f> multiplies the profile's knot times by f (schedule
#                           transfer to an asset with another t_end, see `T_END_HSVD32S`)
#   _dt<t>                  decode the final forward at this time instead of the last node (mh)
#   _x2<c>                  second-order x1 extrapolation in alpha, `eval_sampler.x1_extrap` = c
#                           (z or mh path)
#   _es<mode>               stratified Gaussian draws across positions, `eval_sampler.eta_strat`
#                           ('std'; mh)
#   _scw<w>                 distilled self-conditioning guidance at scale w on an `algo.sc_guidance` model
#                           (`sampling.sc_guidance_scale`); works with any
#                           grid / solver token set
#   _mid                    midpoint node placement, `eval_sampler.grid_midpoint: true`
GRID_RE = re.compile(r'^(?P<base>.+)_gknots(?P<knots>0\.\d+(?:_0\.\d+)+)_a(?P<a>0\.\d+)_b(?P<b>0\.\d+)$')
MH_RE = re.compile(r'^(?P<rest>.+?)(?:_np(?P<np>[a-z]+\d*)(?:x(?P<npf>\d\.\d+))?)?(?:_dt(?P<dt>0\.\d+))?(?:_x2(?P<x2>[\d.]+))?'
                   r'(?:_es(?P<es>std))?(?:_scw(?P<scw>[\d.]+))?$')
PROFILES = {
    # The frozen schedule's coupled noise profile (flat [t, g, ...] knot list, linear between
    # knots, constant outside): the eta refresh and the m mixing ramp from 0.3 x to 2.0 x the mean
    # rate over t = 0.25 -> 0.36.
    'late2': [0.10, 0.3, 0.25, 0.3, 0.36, 2.0, 0.48, 2.0],
}
# The frozen schedule's absolute times belong to the hsvd32s asset (t_end below). On an asset with another t_end every time is
# scaled by f = t_end(new) / t_end(hsvd32s): the launcher scales the cell name's a / b / dt, and the
# profile token takes the factor as a suffix, `_np<profile>x<f>` (e.g. `nplate2x0.9427`). The knot
# TIMES are multiplied by f, the gain values are unchanged. `x1.0` (or no suffix) is the profile as
# frozen.
T_END_HSVD32S = 0.942549920684621


def scaled_profile(name, f):
    """Return `PROFILES[name]` with its knot times (not gains) multiplied by `f`."""
    prof = list(PROFILES[name])
    if f is not None and f != 1.0:
        prof[0::2] = [round(t * f, 6) for t in prof[0::2]]
    return prof


def cell_spec(name):
    """Parse a cell name (token grammar above) into a dict of `sampling.*` / `eval_sampler.*` settings."""
    mid = name.endswith('_mid')
    if mid:
        name = name[:-4]
    mh = {}
    h = MH_RE.match(name)
    if h and h.group('scw'):
        # Self-conditioning guidance scale (not an mh token: no `state_form` change)
        name = h.group('rest')
        mh['sc_guidance_scale'] = float(h.group('scw'))
    if h and h.group('x2'):
        name = h.group('rest')
        mh['x1_extrap'] = float(h.group('x2'))
    if h and (h.group('np') or h.group('dt') or h.group('es')):
        name = h.group('rest')
        mh['state_form'] = 'mh'
        if h.group('np'):
            prof = scaled_profile(h.group('np'), float(h.group('npf')) if h.group('npf') else None)
            mh['sde_eta_gate'], mh['sde_mix_gate'], mh['sde_gate_normalize'] = prof, list(prof), True
        if h.group('dt'):
            mh['decode_t'] = float(h.group('dt'))
        if h.group('es'):
            mh['eta_strat'] = h.group('es')
    g = GRID_RE.match(name)
    grid = {}
    if g:
        name = g.group('base')
        knots = [float(k) for k in g.group('knots').split('_')]
        if len(knots) % 2:
            raise SystemExit(f'{name}: knots must come in (u, v) pairs')
        if not mid:
            raise SystemExit(f'{name}: the knots grid needs the `_mid` suffix')
        grid = dict(grid='knots', grid_knots=knots, grid_a=float(g.group('a')), grid_b=float(g.group('b')))
    spec = _cell_spec_base(name)
    spec.update(grid)
    spec.update(mh)
    if mid:
        spec['grid_midpoint'] = True
    return spec


def _cell_spec_base(name):
    if name in CORE_CELLS:
        return dict(CORE_CELLS[name])
    # sde<noise>[d<decay>][_T<T>]: optional noise-decay exponent (`sde_noise_decay`, default 1),
    # e.g. sde32d2_T256
    m = re.fullmatch(r'sde(\d+(?:\.\d+)?)(?:d(\d+(?:\.\d+)?))?(?:_T(\d+))?', name)
    if m:
        spec = dict(steps=[int(m.group(3) or 256)], solver='sde', sde_noise_scale=float(m.group(1)))
        if m.group(2):
            spec['sde_noise_decay'] = float(m.group(2))
        return spec
    m = re.fullmatch(r'(euler|heun)_T(\d+)', name)
    if m:
        return dict(steps=[int(m.group(2))], solver=m.group(1))
    m = re.fullmatch(r'mg_a(-?[\d.]+)_b([\d.]+)_(euler|sde[\d.]*(?:d[\d.]+)?)_T(\d+)', name)
    if m:
        solver = m.group(3)
        spec = dict(steps=[int(m.group(4))], momentum_guidance=True, mg_alpha=float(m.group(1)),
                    mg_beta=float(m.group(2)))
        if solver.startswith('sde'):
            sub = _cell_spec_base(solver)  # Reuse the sde<noise>[d..] parser
            spec.update({k: v for k, v in sub.items() if k != 'steps'})
        else:
            spec.update(solver=solver)
        return spec
    raise SystemExit(f'unknown cell {name!r}')


def find_checkpoints(run_glob):
    """Map step -> checkpoint dir (latest run dir wins on duplicates; incomplete dirs skipped)."""
    found = {}
    for run_dir in sorted(glob.glob(os.path.join(REPO, run_glob))):
        for ck in glob.glob(os.path.join(run_dir, 'checkpoints', 'checkpoint-*')):
            if not (os.path.isfile(os.path.join(ck, 'model.safetensors'))
                    and os.path.isfile(os.path.join(ck, 'ema.pt'))):
                continue
            found[int(ck.rsplit('-', 1)[1])] = os.path.abspath(ck)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-config', required=True, help='e.g. relf_s (configs/relf_s.yaml)')
    ap.add_argument('--run-glob', required=True, help="e.g. 'outputs/relf_s_2*'")
    ap.add_argument('--steps', default='all', help="comma list of checkpoint steps, or 'all'")
    ap.add_argument('--cells', default=CORE_DEFAULT)
    ap.add_argument('--eval-batch-size', type=int, default=256)
    ap.add_argument('--num-sample-batches', type=int, default=16)
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'configs'))
    ap.add_argument('--seed', type=int, default=None,
                    help='sampling seed override (default: the training config seed); the run_name '
                         'gets a _s<seed> suffix so fresh-seed confirmations never collide')
    ap.add_argument('--shards', type=int, default=1,
                    help='split each cell into K independent processes of 4096/K samples with seeds 1001..100K '
                         '(for big models whose 4096-sample cell would outlive a preemption window); '
                         'scripts/summarize_evals.py --pool merges them')
    ap.add_argument('--shard-seed-base', type=int, default=1001,
                    help='first shard seed (default 1001 -> seeds 1001..100K); pass e.g. 2001 for a fresh-seed '
                         'repeat of an already-measured sharded cell (run_names carry the seed, so no collision)')
    ap.add_argument('--raw', action='store_true',
                    help='evaluate the raw (non-EMA) weights: eval.disable_ema True, run_name gets a _raw suffix')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.shards > 1:
        assert args.num_sample_batches % args.shards == 0, 'shards must divide num_sample_batches'

    base_path = os.path.join(REPO, 'configs', f'{args.train_config}.yaml')
    ckpts = find_checkpoints(args.run_glob)
    if not ckpts:
        sys.exit(f'no complete checkpoints under {args.run_glob}')
    steps = sorted(ckpts) if args.steps == 'all' else [int(s) for s in args.steps.split(',')]
    written = []
    for step in steps:
        if step not in ckpts:
            print(f'[skip] no complete checkpoint-{step}', file=sys.stderr)
            continue
        shard_plan = ([(None, args.num_sample_batches)] if args.shards == 1
                      else [(args.shard_seed_base + k, args.num_sample_batches // args.shards) for k in range(args.shards)])
        for cell in args.cells.split(','):
          for shard_seed, n_batches in shard_plan:
            seed = shard_seed if shard_seed is not None else args.seed
            spec = cell_spec(cell)
            run_name = f'eval_{args.train_config}_ck{step}_{cell}'
            if n_batches * args.eval_batch_size != 4096:
                run_name += f'_n{n_batches * args.eval_batch_size}'
            if seed is not None:
                run_name += f'_s{seed}'
            if args.raw:
                run_name += '_raw'
            out =os.path.join(args.out_dir, f'{run_name}.yaml')
            if os.path.exists(out) and not args.force:
                print(f'[keep] {out}')
                written.append(out)
                continue
            cfg = OmegaConf.load(base_path)
            cfg.mode = 'sample_eval'
            cfg.run_name = run_name
            if seed is not None:
                cfg.seed = seed
            cfg.eval.checkpoint_path = ckpts[step]
            cfg.eval.disable_ema = args.raw
            cfg.loader.eval_batch_size = args.eval_batch_size
            cfg.sampling.num_sample_batches = n_batches
            cfg.sampling.grid = spec.pop('grid', 'uniform')
            cfg.sampling.steps = spec.pop('steps')
            cfg.sampling.sc_guidance_scale = spec.pop('sc_guidance_scale', 1.0)
            # Every other knob lands in `eval_sampler`, whose base-yaml values are the defaults
            for k, v in spec.items():
                if k not in cfg.eval_sampler:
                    raise SystemExit(f'{cell}: unknown eval_sampler key {k}')
                cfg.eval_sampler[k] = v
            header = (
                f'# {run_name}: sample_eval cell "{cell}" on {args.train_config} checkpoint-{step}\n'
                f'# ({ckpts[step]}). Generated by scripts/make_eval_configs.py from configs/'
                f'{args.train_config}.yaml: only mode, run_name, eval.checkpoint_path, sampling.*, eval_sampler.* and\n'
                f'# loader.eval_batch_size differ. EMA weights, {n_batches} x '
                f'{args.eval_batch_size} = {n_batches * args.eval_batch_size} samples per step count'
                f'{" (shard seed " + str(seed) + " of a " + str(args.shards) + "-shard 4096-sample cell)" if shard_seed else ""}, '
                f'single process. Launch bare: python -m main --config-name {run_name}\n'
            )
            with open(out, 'w') as f:
                f.write(header)
                f.write(OmegaConf.to_yaml(cfg, resolve=False))
            print(f'[write] {out}')
            written.append(out)
    names = ' '.join(os.path.basename(w)[:-5] for w in written)
    print(f'\n{len(written)} configs. Run each with: python -m main --config-name <name>')
    print('ALL:', names)


if __name__ == '__main__':
    main()
