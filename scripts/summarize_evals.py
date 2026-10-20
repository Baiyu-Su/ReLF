"""Tabulate sample_eval results from outputs/eval_*/samples.json.

Each eval run folder holds the resolved config.yaml (written by main._print_config) and samples.json
with per_step_metrics {T: {generative_ppl, entropy}}. Rows: train run, checkpoint step, cell,
solver / SDE / MG / carry settings, T, sample count, seed, gen_ppl, entropy.

    python scripts/summarize_evals.py                       # everything under outputs/eval_*
    python scripts/summarize_evals.py --md eval_table.md --csv eval_table.csv
    python scripts/summarize_evals.py --only "eval_relf_s_ck80000_sde eval_relf_s_ck80000_tsweep"
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_RE = re.compile(r'^(?P<run>eval_(?P<train>.+?)_ck(?P<step>\d+)_(?P<cell>.+?)(?:_n\d+)?(?:_s\d+)?)_\d{8}_\S+$')
import math


def rows_for(samples_path):
    """Return one table row per sampling step count T of an eval run's samples.json."""
    run_dir = os.path.dirname(samples_path)
    m = RUN_RE.match(os.path.basename(run_dir))
    cfg_path = os.path.join(run_dir, 'config.yaml')
    cfg = OmegaConf.load(cfg_path) if os.path.isfile(cfg_path) else None
    data = json.load(open(samples_path))
    per_step = data.get('per_step_metrics') or {}
    if not per_step:  # Pre-T-major harness: single pooled pair
        steps = (cfg.sampling.steps if cfg is not None else ['?'])
        per_step = {str(steps[-1]): {'generative_ppl': data['generative_ppl'], 'entropy': data['entropy']}}
    s = cfg.sampling if cfg is not None else {}
    e = cfg.eval_sampler if cfg is not None else {}
    n_samples = (cfg.loader.eval_batch_size * s.num_sample_batches) if cfg is not None else len(data.get('generated_seqs', []))
    solver = e.get('solver', 'euler') if cfg is not None else '?'
    extra = []
    if solver == 'sde':
        extra.append(f"noise={e.get('sde_noise_scale', 1.0)}")
        if float(e.get('sde_noise_decay', 1.0)) != 1.0:
            extra.append(f"decay={e.get('sde_noise_decay')}")
    if e.get('momentum_guidance', False) if cfg is not None else False:
        extra.append(f"mg a={e.mg_alpha} b={e.mg_beta}")
    if (e.get('carry_mode', 'warm') if cfg is not None else 'warm') != 'warm':
        extra.append(f"carry={e.carry_mode}")
    if cfg is not None and float(s.get('sc_guidance_scale', 1.0)) != 1.0:
        extra.append(f"scw={s.sc_guidance_scale}")
    grid = s.get('grid', 'uniform') if cfg is not None else 'uniform'
    if grid == 'knots':
        extra.append(f"grid=knots({','.join(str(float(k)) for k in e.get('grid_knots'))})")
    elif grid != 'uniform':
        extra.append(f"grid={grid}")
    if cfg is not None and e.get('grid_a', None) is not None:
        extra.append(f"ab=({float(e.grid_a):.4f},{float(e.grid_b):.4f})")
    if cfg is not None and bool(e.get('grid_midpoint', False)):
        extra.append('mid')
    if cfg is not None and float(e.get('x1_extrap', 0.0)):
        extra.append(f"x2={e.get('x1_extrap')}")
    if cfg is not None and e.get('state_form', 'z') == 'mh':
        g = lambda k: ('' if e.get(k) is None else ' ' + k[4:7] + 'g' + ','.join(str(float(v)) for v in e.get(k)))  # noqa: E731
        extra.append(f"mh{g('sde_eta_gate')}{g('sde_mix_gate')}{' norm' if e.get('sde_gate_normalize') else ''}"
                     f"{' dec=' + str(e.decode_t) if e.get('decode_t') is not None else ''}"
                     f"{' es=' + str(e.eta_strat) if str(e.get('eta_strat', 'off')) != 'off' else ''}")
    out = []
    for T, met in sorted(per_step.items(), key=lambda kv: int(kv[0])):
        out.append(dict(
            run=m.group('run') if m else os.path.basename(run_dir),
            train=m.group('train') if m else '?', ckpt=int(m.group('step')) if m else -1,
            cell=m.group('cell') if m else '?', solver=solver, settings=' '.join(extra) or '-',
            T=int(T), n_samples=n_samples, seed=(cfg.seed if cfg is not None else '?'),
            ema=(not cfg.eval.disable_ema) if cfg is not None else '?',
            gen_ppl=met['generative_ppl'], entropy=met['entropy'], path=samples_path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='outputs/eval_*/samples.json')
    ap.add_argument('--only', default='', help='space-separated run_name prefixes to include')
    ap.add_argument('--md', default='', help='write a markdown table here')
    ap.add_argument('--csv', default='', help='write CSV here')
    ap.add_argument('--pool', action='store_true',
                    help='merge shard rows (same train/ckpt/cell/T, different seeds): gen_ppl pooled in the '
                         'log domain weighted by n_samples, entropy weighted mean; seed column = pooled(k)')
    args = ap.parse_args()
    only = args.only.split()
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, args.glob))):
        try:
            rs = rows_for(p)
        except Exception as e:  # noqa: BLE001
            print(f'[warn] {p}: {e}', file=sys.stderr)
            continue
        rows += [r for r in rs if not only or any(r['run'].startswith(o) for o in only)]
    if args.pool:
        groups = {}
        for r in rows:
            groups.setdefault((r['train'], r['ckpt'], r['cell'], r['T']), []).append(r)
        pooled = []
        for key, rs in groups.items():
            if len(rs) == 1:
                pooled.append(rs[0]); continue
            n = sum(r['n_samples'] for r in rs)
            lp = sum(r['n_samples'] * math.log(r['gen_ppl']) for r in rs) / n
            ent = sum(r['n_samples'] * r['entropy'] for r in rs) / n
            base = dict(rs[0]); base.update(n_samples=n, seed=f'pooled({len(rs)})', gen_ppl=math.exp(lp), entropy=ent)
            pooled.append(base)
        rows = pooled
    rows.sort(key=lambda r: (r['train'], r['ckpt'], r['cell'], r['T']))
    cols = ['train', 'ckpt', 'cell', 'solver', 'settings', 'T', 'n_samples', 'seed', 'ema', 'gen_ppl', 'entropy']
    fmt = lambda r, c: (f"{r[c]:.2f}" if c == 'gen_ppl' else f"{r[c]:.3f}" if c == 'entropy' else str(r[c]))
    width = {c: max(len(c), *(len(fmt(r, c)) for r in rows)) for c in cols} if rows else {c: len(c) for c in cols}
    print(' '.join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print(' '.join(fmt(r, c).ljust(width[c]) for c in cols))
    if not rows:
        print('(no results)')
    if args.md:
        with open(args.md, 'w') as f:
            f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
            for r in rows:
                f.write('| ' + ' | '.join(fmt(r, c) for c in cols) + ' |\n')
        print('wrote', args.md)
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols + ['path'])
            w.writeheader()
            w.writerows(rows)
        print('wrote', args.csv)


if __name__ == '__main__':
    main()
