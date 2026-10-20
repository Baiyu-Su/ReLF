"""Attach the conditional-channel time schedule to a frozen embedding asset.

Writes `blob['schedule']`, which the `arc_length` t-sampling density and the `arc_length` /
`knots` inference grids read:

  `alpha_of_u` : (lut_size,) fp64 -- monotone u -> alpha lookup on a uniform u grid, u the
               normalized information arc length of (r_c, r_u) below. t = alpha / (1 + alpha);
               storing alpha keeps 1 - t = 1/(1 + alpha) exact in fp64 (the sampler's
               contraction factors need it).
  `alpha_end` / `t_end` : smallest SNR meeting BOTH endpoint tolerances.
  `grid_alpha`, `r`, `s`, `u` : the isotonic-regressed curves for the record.

The information curves:

  r_c(alpha)  bigram-conditional information rate, E[log p(a | y_alpha, h)] - E[log Q_h(a)],
              normalized: the transition band where the classifier resolves tokens GIVEN the
              left neighbor. Pairs (h, a) come from the held-out EVAL slice of the same strided
              chunks the asset was built from; the prior Q_h is the SHAPE-slice `PairTable` at
              the asset's own `tau_bigram`.
  r_u(alpha)  unigram information rate on the same (a, eps) bank: the commit tail -- u keeps
              allocating steps past the conditional band until context-free identification
              completes at the eps-gated alpha_end (`endpoint_alpha`, unigram prior -- the
              decode-at-t_end guarantee must not assume reliable context).

The count tables are rebuilt deterministically from the embedding run's own config and seed (same
draw order as `train_embedding.main`) and checked against the asset's stored `pi`. The embedding
tensor is untouched: the output is a NEW asset file (`cfg.out_asset`) in the run dir, mirrored to
assets/ -- the input asset is never modified (a run may be training on it).

  python scripts/build_schedule.py --config-name <schedule config>   # configs/<name>.yaml
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.train_embedding import (  # noqa: E402
    PairTable,
    count_adjacent,
    endpoint_alpha,
    load_owt_split,
    strided_chunks,
)


def pava_increasing(y):
    """Fit a non-decreasing pool-adjacent-violators isotonic regression to a 1-D sequence.

    Returns an fp64 CPU tensor of the same length.
    """
    vals = [float(v) for v in y]
    w = [1.0] * len(vals)
    out_v, out_w = [], []
    for v, ww in zip(vals, w):
        out_v.append(v)
        out_w.append(ww)
        while len(out_v) > 1 and out_v[-2] > out_v[-1]:
            v2, w2 = out_v.pop(), out_w.pop()
            v1, w1 = out_v.pop(), out_w.pop()
            out_v.append((v1 * w1 + v2 * w2) / (w1 + w2))
            out_w.append(w1 + w2)
    res = []
    for v, ww in zip(out_v, out_w):
        res.extend([v] * int(ww))
    return torch.tensor(res, dtype=torch.float64)


@torch.no_grad()
def rate_curves(E, sq, logq_rows, log_prior_a, a, eps, grid, chunk):
    """Compute r(alpha_g) = E[log p(a | y_alpha, prior)] - E[log prior(a)] on a fixed bank, CRN.

    `logq_rows` is either (n, V) fp32 per-sample log prior rows (conditional) or a (V,) fp32
    vector (unigram, broadcast). Per chunk the alpha-independent pieces G_a = E[a] @ E^T and
    N_e = eps @ E^T are computed once and swept over the whole grid:
      logit_b = log prior(b) + alpha^2 G_a[b] + alpha N_e[b] - (alpha^2 / 2) sq[b].
    Returns a (len(grid),) fp64 tensor.
    """
    dev = eps.device
    G = len(grid)
    acc = torch.zeros(G, dtype=torch.float64, device=dev)
    n = len(a)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(s0 + chunk, n))
        ai = a[sl].long()
        Ga = E[ai] @ E.T  # (m, V) fp32
        Ne = eps[sl] @ E.T
        lq = logq_rows if logq_rows.dim() == 1 else logq_rows[sl]
        ar = torch.arange(len(ai), device=dev)
        for g, al in enumerate(grid):
            al = float(al)
            logits = lq + (al * al) * Ga + al * Ne - (0.5 * al * al) * sq
            acc[g] += logits.log_softmax(dim=-1)[ar, ai].double().sum()
    return acc / n - float(log_prior_a.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-name', required=True, help='schedule config in configs/')
    args = ap.parse_args()
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, 'configs', f'{args.config_name}.yaml'))
    ecfg = OmegaConf.load(os.path.join(REPO_ROOT, 'configs', f'{cfg.embed_config}.yaml'))

    t0 = time.time()
    assert torch.cuda.is_available()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False

    job = os.environ.get('SLURM_JOB_ID') or os.environ.get('TORCHELASTIC_RUN_ID')
    job = job or datetime.datetime.now().strftime('%H%M%S')
    run_dir = os.path.join(
        REPO_ROOT, 'outputs', f'{cfg.run_name}_{datetime.datetime.now():%Y%m%d}_{job}'
    )
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'resolved_config.yaml'))
    print(f'[run] dir={run_dir} device={torch.cuda.get_device_name()}', flush=True)

    asset_path = os.path.join(REPO_ROOT, cfg.asset_path)
    blob = torch.load(asset_path, map_location='cpu', weights_only=False)
    assert 'schedule' not in blob, 'asset already carries a schedule'
    V, d = ecfg.vocab_size, ecfg.flow_dim
    E = blob['embedding'][:V].float().to(dev)  # Drop the zero [PAD] row
    sq = (E * E).sum(1)
    tau = float(blob['tau_bigram'])

    # Deterministic SHAPE/EVAL slice rebuild (same seed/draw order as `train_embedding`)
    cpu_gen = torch.Generator().manual_seed(ecfg.seed + 1)
    gpu_gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    train_ds = load_owt_split(
        'openwebtext-train', 'train', ecfg.data.cache_dir, ecfg.data.block_size
    )
    shape_chunks, eval_chunks = [], []
    for i, rows in enumerate(
        strided_chunks(train_ds, ecfg.data.stats_windows, ecfg.data.stats_chunks, cpu_gen)
    ):
        tok = rows.to(dev, non_blocking=True).long()
        (eval_chunks if i % 4 == 3 else shape_chunks).append(tok)
    keys, cnts = count_adjacent(shape_chunks, V)
    tab = PairTable(keys, cnts, V)
    pi_err = float((tab.pi.cpu() - blob['pi'].double()).abs().max())
    # `blob['pi']` is stored fp32 and the rebuild is fp64, so the default tolerance is fp32
    # rounding, not exactness. An asset whose pi was copied from a build on another copy of the
    # OWT cache can set `pi_tol` in its schedule yaml: the schedule then uses this cache's SHAPE
    # tables and the training run the asset's pi -- a 1e-5-level difference.
    pi_tol = float(cfg.get('pi_tol', 1e-7))
    assert pi_err < pi_tol, f'rebuilt SHAPE-slice pi does not match the asset (max diff {pi_err}, pi_tol {pi_tol})'
    if pi_err >= 1e-7:
        print(f'[counts] NOTE: asset pi differs from this cache\'s SHAPE slice by {pi_err:.3e} (allowed by pi_tol={pi_tol:g})', flush=True)
    print(f'[counts] SHAPE tables rebuilt + checksummed vs blob pi (tau={tau:g})', flush=True)

    # EVAL-slice adjacent-pair bank (held out from the prior estimation), CRN eps
    ha = torch.cat([t[:, :-1].reshape(-1) for t in eval_chunks])
    aa = torch.cat([t[:, 1:].reshape(-1) for t in eval_chunks])
    m = (ha < V) & (aa < V)
    ha, aa = ha[m], aa[m]
    sel = torch.randperm(len(ha), device=dev, generator=gpu_gen)[: cfg.pair_bank]
    ha, aa = ha[sel], aa[sel]
    del shape_chunks, eval_chunks, keys, cnts
    torch.cuda.empty_cache()
    eps = torch.randn(cfg.pair_bank, d, device=dev, generator=gpu_gen)
    logq_rows = torch.empty(cfg.pair_bank, V, dtype=torch.float32, device=dev)
    for s0 in range(0, cfg.pair_bank, cfg.chunk):
        sl = slice(s0, min(s0 + cfg.chunk, cfg.pair_bank))
        logq_rows[sl] = tab.dense_rows(ha[sl], tau).log().float()
    ar = torch.arange(cfg.pair_bank, device=dev)
    log_prior_c = logq_rows[ar, aa.long()]
    logpi = torch.where(
        tab.pi > 0, tab.pi.log(), torch.tensor(-1e30, dtype=torch.float64, device=dev)
    ).float()
    HA = float(-(tab.pi[tab.pi > 0] * tab.pi[tab.pi > 0].log()).sum())
    Hc = float(-log_prior_c.mean())
    print(f'[bank] {cfg.pair_bank} EVAL pairs; H(A)={HA:.4f}  H(A|prev)~={Hc:.4f} nats', flush=True)

    # Endpoint alpha: unigram prior, eps gates (decode-at-t_end must not assume context)
    cum = tab.pi.cumsum(0)
    u0 = torch.rand(cfg.endpoint_n, dtype=torch.float64, device=dev, generator=gpu_gen) * cum[-1]
    end_a = torch.searchsorted(cum, u0).clamp_max_(V - 1)
    end_eps = torch.randn(cfg.endpoint_n, d, device=dev, generator=gpu_gen)
    a_end, (ent_end, err_end) = endpoint_alpha(
        E, logpi, sq, end_a, end_eps, HA, cfg.eps_H, cfg.eps_err, cfg.alpha_cap, 4096, iters=24
    )
    assert math.isfinite(a_end), 'endpoint tolerances unreachable below alpha_cap'
    t_end = a_end / (1.0 + a_end)
    print(
        f'[endpoint] alpha_end={a_end:.3f} t_end={t_end:.5f} entH={ent_end:.2e} err={err_end:.2e}',
        flush=True,
    )

    # Conditional + unigram rate curves on a shared log-alpha grid, CRN across the grid
    grid = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64),
            torch.logspace(math.log10(cfg.alpha_min), math.log10(a_end), cfg.grid_j, dtype=torch.float64),
        ]
    )
    R_c = rate_curves(E, sq, logq_rows, log_prior_c, aa, eps, grid, cfg.chunk)
    print(f'[curve] conditional: R_c(alpha_end)={float(R_c[-1]):.4f} nats ({time.time() - t0:.0f}s)', flush=True)
    del logq_rows
    torch.cuda.empty_cache()
    n_u = min(cfg.rate_bank, cfg.pair_bank)
    R_u = rate_curves(
        E, sq, logpi, logpi[aa[:n_u].long()], aa[:n_u], eps[:n_u], grid, cfg.chunk
    )
    print(f'[curve] unigram: R_u(alpha_end)={float(R_u[-1]):.4f} nats ({time.time() - t0:.0f}s)', flush=True)

    r_c = pava_increasing((R_c / R_c[-1]).clamp(0.0, 1.0).cpu())
    r_u = pava_increasing((R_u / R_u[-1]).clamp(0.0, 1.0).cpu())
    r_c[0] = r_u[0] = 0.0
    r_c = (r_c / r_c[-1]).clamp(0.0, 1.0)
    r_u = (r_u / r_u[-1]).clamp(0.0, 1.0)

    # Information arc length u of (r_c, r_u) and the uniform-u alpha lookup
    dl = ((r_c[1:] - r_c[:-1]) ** 2 + (r_u[1:] - r_u[:-1]) ** 2).sqrt()
    u = torch.cat([torch.zeros(1, dtype=torch.float64), dl.cumsum(0)])
    u = u + torch.arange(len(u), dtype=torch.float64) * 1e-12  # Strictly increasing
    u = u / u[-1]
    uq = torch.linspace(0.0, 1.0, cfg.lut_size, dtype=torch.float64)
    idx = torch.searchsorted(u, uq).clamp_(1, len(u) - 1)
    w = ((uq - u[idx - 1]) / (u[idx] - u[idx - 1])).clamp(0.0, 1.0)
    alpha_of_u = torch.lerp(grid[idx - 1], grid[idx], w)
    alpha_of_u[0], alpha_of_u[-1] = 0.0, float(a_end)
    assert bool((alpha_of_u[1:] >= alpha_of_u[:-1]).all()), 'alpha_of_u must be monotone'

    schedule = {
        'alpha_of_u': alpha_of_u,  # fp64, uniform u grid, alpha_of_u[-1] = alpha_end
        'alpha_end': float(a_end),
        't_end': float(t_end),
        'eps_H': float(cfg.eps_H),
        'eps_err': float(cfg.eps_err),
        'ent_end': float(ent_end),
        'err_end': float(err_end),
        'grid_alpha': grid,
        'r': r_c,
        's': r_u,
        'u': u,
        'coordinate': 'information arc length of (r_cond, r_unigram)',
        'H_A': HA,
        'H_A_given_prev': Hc,
        'R_c_end': float(R_c[-1]),
        'R_u_end': float(R_u[-1]),
        'built': f'{datetime.datetime.now():%Y-%m-%d} job {job}',
        'config': OmegaConf.to_yaml(cfg),
    }
    blob['schedule'] = schedule
    out_run = os.path.join(run_dir, cfg.out_asset)
    torch.save(blob, out_run)
    repo_asset = os.path.join(REPO_ROOT, 'assets', cfg.out_asset)
    torch.save(blob, repo_asset)

    summary = {k: (v if not torch.is_tensor(v) else 'tensor') for k, v in schedule.items()}
    summary.update({'in_asset': asset_path, 'out': [out_run, repo_asset],
                    'total_min': round((time.time() - t0) / 60, 1)})
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
    tq = (alpha_of_u / (1 + alpha_of_u))[(q * (cfg.lut_size - 1)).long()]
    print(
        f'[done] t at u=(.1 .25 .5 .75 .9): {[round(float(x), 4) for x in tq]}; '
        f'alpha_end={a_end:.2f} t_end={t_end:.5f}; wrote {out_run} + {repo_asset}',
        flush=True,
    )


if __name__ == '__main__':
    main()
