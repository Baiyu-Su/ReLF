"""Build the ReLF flow-space embedding E from OpenWebText co-occurrence statistics.

Pipeline (single GPU, corpus-only, no neural network):
  1. Adjacent-pair count tables on disjoint SHAPE / EVAL slices of the cached OWT windows;
     bigram smoothing tau selected by held-out (OWT-valid) next-token log loss.
  2. Init: Hellinger-SVD of the both-neighbor context distributions (`ContextStats` +
     `hellinger_svd_init`; the dense V x C Hellinger matrix is never formed), duplicate-row
     jitter, then the pi-weighted gauge (zero mean, average power P).
  3. `mode`:
       hellinger_svd   export the init as is (the shipped assets);
       svd_collision   Adam steps on the collision barrier only (pushes near-duplicates apart);
       force_field     commit-cost force field on the top-N confusable universe: pairs are
                       repelled through K(d) = sum_sigma exp(-d^2 / 2 sigma^2) in proportion to
                       their conditional substitution cost c(a, b) (SHAPE tables, bf16 dense
                       cache), substitutable pairs (c < c0) weakly attracted, floored at d_att.
  4. Every asset records the held-out EVAL-slice nearest-neighbor substitution-cost audit, the
     endpoint identifiability gate (alpha_end) and a permutation control.

Export: `blob['embedding']` of shape (50258, d) with a zero [PAD] row appended last, plus the
unigram `pi`; the gauge is pi-weighted zero mean and pi-weighted power P = d. Do NOT rescale it --
`scripts/build_schedule.py` computes the paired time schedule at exactly this scale.

Usage:
    python scripts/train_embedding.py --config-name <embed config>   # configs/<name>.yaml
    python scripts/train_embedding.py --self-test
    python scripts/train_embedding.py --smoke   # tiny debugging run
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


# Data


def load_owt_split(name, mode, cache_dir, block_size):
    """Load a cached OWT split; refuse to rebuild it here."""
    from relf import dataloader

    fname = f'{name}_{mode}_bs{block_size}_wrapped.dat'
    path = os.path.join(cache_dir, fname)
    assert os.path.isdir(path), f'missing cached dataset {path}; refusing to rebuild here'
    return dataloader.get_dataset(
        name,
        tokenizer=None,
        mode=mode,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=8,
    )


def strided_chunks(ds, n_windows, n_chunks, cpu_gen):
    """Yield `n_chunks` jittered, evenly strided slices of `input_ids` (`n_windows` in total)."""
    N = len(ds)
    n_windows = min(n_windows, N)
    per = n_windows // n_chunks
    gap = N // n_chunks
    assert per >= 1 and gap >= per, 'stats slice larger than dataset'
    for i in range(n_chunks):
        jitter = int(torch.randint(max(gap - per, 1), (1,), generator=cpu_gen))
        s0 = i * gap + jitter
        yield ds[s0 : s0 + per]['input_ids']


def sample_probe_pairs(tok, n, R, V, gen):
    """Sample `n` probe pairs (a, c) straight from windows (statistics-validation split)."""
    dev = tok.device
    n_w, L = tok.shape
    m = int(1.25 * n) + 64
    wi = torch.randint(n_w, (m,), device=dev, generator=gen)
    pos = torch.randint(L, (m,), device=dev, generator=gen)
    off = torch.randint(2 * R, (m,), device=dev, generator=gen)
    rel = off - R + (off >= R).long()
    j = pos + rel
    ok = (j >= 0) & (j < L)
    wi, pos, off, j = wi[ok][:n], pos[ok][:n], off[ok][:n], j[ok][:n]
    assert len(wi) == n, 'probe-pair oversampling too small'
    a = tok[wi, pos]
    c = off * V + tok[wi, j]
    return a.int(), c.int()



# Counts


def merge_counts(k1, c1, k2, c2):
    """Merge two (sorted key, int32 count) sets, summing the counts of shared keys."""
    k = torch.cat([k1, k2])
    c = torch.cat([c1, c2])
    del k1, k2
    k, order = k.sort()
    c = c[order]
    del order
    new = torch.ones_like(k, dtype=torch.bool)
    new[1:] = k[1:] != k[:-1]
    seg = new.cumsum(0) - 1
    out = torch.zeros(int(seg[-1].item()) + 1, dtype=torch.long, device=k.device)
    out.index_add_(0, seg, c.long())
    return k[new], out.to(torch.int32)



def count_both_neighbors(tok_chunks, V):
    """Count (token, neighbor) pairs for the R=1 both-neighbor `ContextStats`.

    Context id `c = off_idx * V + b` with off_idx 0 = left neighbor, 1 = right neighbor; pairs
    touching an out-of-vocab token are dropped. Returns sorted keys `a * 2V + c` with int32 counts.
    """
    C_dim = 2 * V
    keys = cnts = None
    for tok in tok_chunks:
        a_l, b_l = tok[:, 1:].reshape(-1), tok[:, :-1].reshape(-1)
        m = (a_l < V) & (b_l < V)
        a_l, b_l = a_l[m], b_l[m]
        a_r, b_r = tok[:, :-1].reshape(-1), tok[:, 1:].reshape(-1)
        m = (a_r < V) & (b_r < V)
        a_r, b_r = a_r[m], b_r[m]
        k = torch.cat([a_l * C_dim + b_l, a_r * C_dim + V + b_r]).sort().values
        k, c = torch.unique_consecutive(k, return_counts=True)
        c = c.to(torch.int32)
        keys, cnts = (k, c) if keys is None else merge_counts(keys, cnts, k, c)
    return keys, cnts

def count_adjacent(tok_chunks, V):
    """Count (cur, next) pairs over packed windows as sorted sparse keys `a * V + b`.

    Pairs touching an out-of-vocab token (stray [PAD] in the cache) are dropped.
    """
    keys = cnts = None
    for tok in tok_chunks:
        a = tok[:, :-1].reshape(-1)
        b = tok[:, 1:].reshape(-1)
        m = (a < V) & (b < V)
        k = (a[m] * V + b[m]).sort().values
        k, c = torch.unique_consecutive(k, return_counts=True)
        c = c.to(torch.int32)
        keys, cnts = (k, c) if keys is None else merge_counts(keys, cnts, k, c)
    return keys, cnts


class ContextStats:
    """Sorted-key sparse N[a, c] plus the derived pi, N_a, floored q_C."""

    def __init__(self, keys, cnts, V, R, qc_floor):
        dev = keys.device
        self.V, self.R, self.C_dim = V, R, 2 * R * V
        self.keys, self.cnt = keys, cnts
        self.row = torch.div(keys, self.C_dim, rounding_mode='floor').to(torch.int32)
        self.col = (keys - self.row.long() * self.C_dim).to(torch.int32)
        cf = cnts.double()
        self.Na = torch.zeros(V, dtype=torch.float64, device=dev)
        self.Na.index_add_(0, self.row.long(), cf)
        self.N = float(cf.sum())
        self.pi = self.Na / self.N
        col_cnt = torch.zeros(self.C_dim, dtype=torch.float64, device=dev)
        col_cnt.index_add_(0, self.col.long(), cf)
        self.qc = (col_cnt + qc_floor) / (self.N + qc_floor * self.C_dim)
        p = self.pi[self.pi > 0]
        self.HA = float(-(p * p.log()).sum())

    def nac(self, a, c):
        q = a.long() * self.C_dim + c.long()
        pos = torch.searchsorted(self.keys, q).clamp_max_(len(self.keys) - 1)
        return torch.where(
            self.keys[pos] == q,
            self.cnt[pos].double(),
            torch.zeros((), dtype=torch.float64, device=q.device),
        )

    def select_tau(self, va, vc, grid):
        """Pick the tau in `grid` with the lowest held-out context log loss on (`va`, `vc`)."""
        nac = self.nac(va, vc)
        na = self.Na[va.long()]
        qcc = self.qc[vc.long()]
        losses = {}
        for tau in grid:
            losses[float(tau)] = float(-((nac + tau * qcc).log() - (na + tau).log()).mean())
        best = min(losses, key=losses.get)
        return best, losses

    def iac(self, tau, chunk=1 << 26):
        """Return the exact I(A;C) under the smoothed Q.

        The support term is chunked; the off-support term is closed form.
        """
        dev = self.keys.device
        T = torch.zeros(self.V, dtype=torch.float64, device=dev)
        S = torch.zeros(self.V, dtype=torch.float64, device=dev)
        for s0 in range(0, len(self.keys), chunk):
            r = self.row[s0 : s0 + chunk].long()
            qcc = self.qc[self.col[s0 : s0 + chunk].long()]
            Q = (self.cnt[s0 : s0 + chunk].double() + tau * qcc) / (self.Na[r] + tau)
            T.index_add_(0, r, Q * (Q / qcc).log())
            S.index_add_(0, r, qcc)
        shr = tau / (self.Na + tau)
        off = shr * shr.log() * (1.0 - S).clamp_min(0.0)
        return float((self.pi * (T + off)).sum())


class PairTable:
    """Smoothed row-conditionals Q_a(b) = (N_ab + tau q0_b) / (N_a + tau) of a (V, V) table."""

    def __init__(self, keys, cnts, V):
        self.V = V
        self.keys, self.cnt = keys, cnts.double()
        self.row = torch.div(keys, V, rounding_mode='floor')
        self.col = keys - self.row * V
        dev = keys.device
        self.Na = torch.zeros(V, dtype=torch.float64, device=dev)
        self.Na.index_add_(0, self.row, self.cnt)
        self.N = float(self.cnt.sum())
        self.pi = self.Na / self.N
        Nb = torch.zeros(V, dtype=torch.float64, device=dev)
        Nb.index_add_(0, self.col, self.cnt)
        self.q0 = (Nb + 0.5) / (self.N + 0.5 * V)

    def transpose(self):
        keysT = self.col * self.V + self.row
        order = keysT.argsort()
        return PairTable(keysT[order], self.cnt[order].to(torch.int32), self.V)

    def lookup(self, a, b):
        q = a * self.V + b
        pos = torch.searchsorted(self.keys, q).clamp_max_(len(self.keys) - 1)
        zero = torch.zeros((), dtype=torch.float64, device=q.device)
        return torch.where(self.keys[pos] == q, self.cnt[pos], zero)

    def select_tau(self, a, b, grid):
        losses = {}
        for tau in grid:
            nab = self.lookup(a, b)
            losses[float(tau)] = float(
                -((nab + tau * self.q0[b]).log() - (self.Na[a] + tau).log()).mean()
            )
        best = min(losses, key=lambda t: losses[t])
        return best, losses

    def csc_struct(self):
        order = torch.argsort(self.col, stable=True)
        ccol = torch.zeros(self.V + 1, dtype=torch.long, device=self.keys.device)
        ccol[1:] = torch.bincount(self.col, minlength=self.V).cumsum(0)
        return {'ccol': ccol, 'row': self.row[order], 'cnt': self.cnt[order].float()}

    def _scatter_dense(self, starts, lens, cols, vals, n, base):
        dev = self.keys.device
        out = base.contiguous()
        sid = torch.repeat_interleave(torch.arange(n, device=dev), lens)
        flat = torch.repeat_interleave(starts, lens) + (
            torch.arange(int(lens.sum()), device=dev)
            - torch.repeat_interleave(lens.cumsum(0) - lens, lens)
        )
        out[sid, cols[flat]] += vals[flat]
        return out

    def dense_rows(self, a, tau):
        """Return the (n, V) smoothed rows Q_a(.) as float64."""
        st = torch.searchsorted(self.keys, a * self.V)
        en = torch.searchsorted(self.keys, (a + 1) * self.V)
        base = (self.q0[None, :] * tau).expand(len(a), self.V)
        out = self._scatter_dense(st, en - st, self.col, self.cnt, len(a), base)
        return out / (self.Na[a] + tau)[:, None]

    def dense_cols(self, b, csc, tau):
        """Return the (n, V) smoothed columns Q_.(b_j) over all rows as float32."""
        st, en = csc['ccol'][b], csc['ccol'][b + 1]
        base = (tau * self.q0[b].float())[:, None].expand(len(b), self.V)
        out = self._scatter_dense(st, en - st, csc['row'], csc['cnt'], len(b), base)
        return out / (self.Na + tau).float()[None, :]


# Hellinger-SVD init


def hellinger_svd_init(stats, tau, d, dev, gen, oversample=16, n_iter=2, chunk=1 << 26):
    """Return the (V, d) Hellinger-SVD init, rows (h_a - hbar) V_d.

    H[a, c] = sqrt(Q_a(c)) = s_a * g_c off-support (+ sparse correction `sval` on support):
    rank-1 + sparse structure, so the randomized SVD of the pi-weighted centered
    B = sqrt(pi) (H - hbar) needs only matvecs and the dense V x C matrix is never formed.
    """
    C = stats.C_dim
    Naf = stats.Na.float()
    qcf = stats.qc.float()
    s_a = (tau / (Naf + tau)).sqrt()
    g = qcf.sqrt()
    pi_f = stats.pi.float()
    sp = pi_f.sqrt()
    nnz = len(stats.keys)
    sval = torch.empty(nnz, dtype=torch.float32, device=dev)
    for s0 in range(0, nnz, chunk):
        r = stats.row[s0 : s0 + chunk].long()
        cc = stats.col[s0 : s0 + chunk].long()
        q = (stats.cnt[s0 : s0 + chunk].float() + tau * qcf[cc]) / (Naf[r] + tau)
        sval[s0 : s0 + chunk] = q.sqrt() - s_a[r] * g[cc]

    def h_mv(Om):  # (C, k) -> (V, k)
        out = s_a[:, None] * (g @ Om)[None, :]
        for s0 in range(0, nnz, chunk):
            r = stats.row[s0 : s0 + chunk].long()
            cc = stats.col[s0 : s0 + chunk].long()
            out.index_add_(0, r, sval[s0 : s0 + chunk, None] * Om[cc])
        return out

    def ht_mv(U):  # (V, k) -> (C, k)
        out = g[:, None] * (s_a @ U)[None, :]
        for s0 in range(0, nnz, chunk):
            r = stats.row[s0 : s0 + chunk].long()
            cc = stats.col[s0 : s0 + chunk].long()
            out.index_add_(0, cc, sval[s0 : s0 + chunk, None] * U[r])
        return out

    hbar = (pi_f * s_a).sum() * g
    for s0 in range(0, nnz, chunk):
        r = stats.row[s0 : s0 + chunk].long()
        cc = stats.col[s0 : s0 + chunk].long()
        hbar.index_add_(0, cc, pi_f[r] * sval[s0 : s0 + chunk])

    def b_mv(Om):
        return sp[:, None] * (h_mv(Om) - (hbar @ Om)[None, :])

    def bt_mv(U):
        U2 = sp[:, None] * U
        return ht_mv(U2) - hbar[:, None] * U2.sum(0)[None, :]

    k = d + oversample
    Om = torch.randn(C, k, device=dev, generator=gen)
    Y = b_mv(Om)
    del Om
    for _ in range(n_iter):
        Y, _ = torch.linalg.qr(Y)
        Y = b_mv(bt_mv(Y))
    Qm, _ = torch.linalg.qr(Y)
    Cm = bt_mv(Qm)  # (C, k); B ~= Qm Cm^T
    evals, U = torch.linalg.eigh(Cm.T @ Cm)
    top = torch.argsort(evals, descending=True)[:d]
    Vd = (Cm @ U[:, top]) / evals[top].clamp_min(1e-30).sqrt()[None, :]
    E0 = h_mv(Vd) - (hbar @ Vd)[None, :]  # Rows (h_a - hbar) V_d
    return E0.contiguous()


# Gaussian channel (endpoint identifiability)


def posterior_logp(E, logpi, sq, a, alpha_row, eps):
    """Return log p_E(. | y, alpha) with y = alpha e_a + eps.

    The -||y||^2/2 term is dropped (softmax-invariant).
    """
    y = alpha_row[:, None] * E[a] + eps
    logits = alpha_row[:, None] * (y @ E.T)
    logits = logits - 0.5 * alpha_row.square()[:, None] * sq[None, :] + logpi[None, :]
    return logits.log_softmax(dim=1)



@torch.no_grad()
def rate_stats(E, logpi, sq, a, eps, alpha_row, chunk, want_err=False):
    """Return per-state log p(a|y) and the mmse contribution ||e_a - xhat||^2.

    With `want_err`, also return the posterior entropy and top-1 error (endpoint criteria).
    """
    lp, mm, ent, err = [], [], [], []
    for s0 in range(0, len(a), chunk):
        sl = slice(s0, s0 + chunk)
        ai = a[sl].long()
        logp = posterior_logp(E, logpi, sq, ai, alpha_row[sl], eps[sl])
        ar = torch.arange(logp.shape[0], device=logp.device)
        lp.append(logp[ar, ai])
        p = logp.exp()
        xhat = p @ E
        mm.append((E[ai] - xhat).square().sum(1))
        if want_err:
            ent.append(-(p * logp).sum(1))
            err.append((logp.argmax(1) != ai).float())
    if want_err:
        return torch.cat(lp), torch.cat(mm), torch.cat(ent), torch.cat(err)
    return torch.cat(lp), torch.cat(mm)



@torch.no_grad()
def endpoint_alpha(E, logpi, sq, a, eps, HA, eps_h, eps_err, cap, chunk, iters=16):
    """Bisect in log alpha for the smallest alpha meeting BOTH endpoint tolerances.

    Returns (alpha_end, (entropy / H(A), top-1 error)); alpha_end is inf when `cap` already fails.
    """
    dev = E.device

    def ok(alpha):
        al = torch.full((len(a),), alpha, device=dev)
        _, _, ent, err = rate_stats(E, logpi, sq, a, eps, al, chunk, want_err=True)
        return float(ent.mean()) / HA <= eps_h and float(err.mean()) <= eps_err, (
            float(ent.mean()) / HA,
            float(err.mean()),
        )

    ok_cap, m_cap = ok(cap)
    if not ok_cap:
        return float('inf'), m_cap
    lo, hi = math.log(1e-2), math.log(cap)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        good, _ = ok(math.exp(mid))
        hi, lo = (mid, lo) if good else (hi, mid)
    a_end = math.exp(hi)
    _, m_end = ok(a_end)
    return a_end, m_end

# Constraints


@torch.no_grad()
def project_(E, pi_f, P):
    """Project E in place to pi-weighted zero mean and pi-weighted average power P.

    Never row-normalize.
    """
    E.sub_((pi_f @ E)[None, :])
    pe = (pi_f * E.square().sum(1)).sum()
    E.mul_((P / pe.clamp_min(1e-30)).sqrt())


@torch.no_grad()
def dedup_jitter_(E, delta0, gen):
    """Jitter exact duplicate rows of E in place; return how many rows were jittered.

    Exact duplicate rows are forbidden but arise in the Hellinger init when shrinkage
    makes rare tokens' context rows identical. The collision barrier has zero gradient at exact
    coincidence, so split them here with a jitter that lands duplicate pairs just past delta0
    (pair distance ~ delta0 * sqrt(2)) and let the barrier hold.
    """
    V, d = E.shape
    _, inv, cnt = torch.unique(E, dim=0, return_inverse=True, return_counts=True)
    dup = cnt[inv] > 1
    n = int(dup.sum())
    if n:
        E[dup] += (delta0 / math.sqrt(d)) * torch.randn(n, d, device=E.device, generator=gen)
    return n


@torch.no_grad()
def refresh_nn(E, k, chunk=8192):
    """Return each row's k nearest-neighbor indices and the global min pairwise row distance."""
    V = E.shape[0]
    sq = E.square().sum(1)
    idx = torch.empty(V, k, dtype=torch.long, device=E.device)
    dmin = float('inf')
    for s0 in range(0, V, chunk):
        s1 = min(s0 + chunk, V)
        d2 = sq[s0:s1, None] + sq[None, :] - 2.0 * (E[s0:s1] @ E.T)
        d2[torch.arange(s1 - s0, device=E.device), torch.arange(s0, s1, device=E.device)] = float(
            'inf'
        )
        top = d2.topk(k, dim=1, largest=False)
        idx[s0:s1] = top.indices
        dmin = min(dmin, float(top.values[:, 0].min().clamp_min(0.0)))
    return idx, math.sqrt(max(dmin, 0.0))


def barrier_pens(E, nn_idx, delta0, p_max):
    """Return the collision barrier over kNN pairs closer than delta0 and the norm-cap penalty."""
    d2 = (E[:, None, :] - E[nn_idx]).square().sum(-1)
    col = torch.relu(delta0**2 - d2).square().sum()
    cap = torch.relu(E.square().sum(1) - p_max).square().sum()
    return col, cap


# Commit-cost machinery


def build_cost_cache(tab, tabT, univ, tau, chunk=512):
    """Build a bf16 dense cache over the universe: fwd/prev conditional rows and log-columns."""
    cscF, cscT = tab.csc_struct(), tabT.csc_struct()
    n, V = len(univ), tab.V
    dev = univ.device
    cache = {k: torch.empty(n, V, dtype=torch.bfloat16, device=dev) for k in ['prev', 'fwd', 'colF', 'colP']}
    for s0 in range(0, n, chunk):
        a = univ[s0 : s0 + chunk]
        cache['fwd'][s0 : s0 + chunk] = tab.dense_rows(a, tau).bfloat16()
        cache['prev'][s0 : s0 + chunk] = tabT.dense_rows(a, tau).bfloat16()
        cache['colF'][s0 : s0 + chunk] = tab.dense_cols(a, cscF, tau).clamp_min(1e-38).log().bfloat16()
        cache['colP'][s0 : s0 + chunk] = tabT.dense_cols(a, cscT, tau).clamp_min(1e-38).log().bfloat16()
    return cache


def pair_cost_cached(cache, ia, ib, chunk=1024):
    """Compute the conditional substitution cost c(a, b) of universe-index pairs from the cache.

    c(a,b) = sum_h Prev_a(h)[logP(a|h)-logP(b|h)] + sum_h Fwd_a(h)[logP(a bef h)-logP(b bef h)]
    """
    out = torch.empty(len(ia), device=ia.device)
    for s0 in range(0, len(ia), chunk):
        sl = slice(s0, s0 + chunk)
        pa = cache['prev'][ia[sl]].float()
        fa = cache['fwd'][ia[sl]].float()
        dF = cache['colF'][ia[sl]].float() - cache['colF'][ib[sl]].float()
        dP = cache['colP'][ia[sl]].float() - cache['colP'][ib[sl]].float()
        out[sl] = (pa * dF).sum(1) + (fa * dP).sum(1)
    return out


@torch.no_grad()
def knn_pairs(E, univ, k, chunk=4096):
    Et = E[univ]
    sq = (Et * Et).sum(1)
    n = len(univ)
    idx = torch.empty(n, k, dtype=torch.long, device=E.device)
    dist = torch.empty(n, k, device=E.device)
    for s0 in range(0, n, chunk):
        d2 = sq[s0 : s0 + chunk, None] + sq[None, :] - 2 * Et[s0 : s0 + chunk] @ Et.T
        ar = torch.arange(s0, min(s0 + chunk, n), device=E.device)
        d2[ar - s0, ar] = float('inf')
        top = d2.topk(k, dim=1, largest=False)
        idx[s0 : s0 + chunk] = top.indices
        dist[s0 : s0 + chunk] = top.values.clamp_min(0).sqrt()
    return idx, dist


def spearman(x, y):
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


@torch.no_grad()
def nn_audit(E, eval_cache, eval_univ, k=1):
    """Audit held-out NN substitution cost over the eval universe (EVAL-slice tables)."""
    idx, dist = knn_pairs(E, eval_univ, k)
    ia = torch.arange(len(eval_univ), device=E.device).repeat_interleave(k)
    ib = idx.reshape(-1)
    cost = pair_cost_cached(eval_cache, ia, ib)
    return {
        'median_cost': round(float(cost.median()), 4),
        'mean_cost': round(float(cost.mean()), 4),
        'frac_gt3': round(float((cost > 3.0).float().mean()), 4),
        'spearman': round(spearman(dist.reshape(-1), cost), 4),
    }


# Training


def train_force_field(E0, tab, cache, univ, ev, cfg, log, run_dir):
    """Run the commit-cost force field from `E0` and select the best healthy iterate.

    Selection takes the healthy eval iterate (the init included) with the lowest held-out median
    NN cost, or the final iterate if none is healthy; it is saved to `run_dir/E_best.pt` and
    returned as a dict (`E`, `step`, `metrics`).

    `cfg.mode` 'svd_collision': the same optimizer
    loop with ONLY the collision barrier -- no commit-cost repulsion/attraction, no kNN pair bank,
    no cost cache -- so the result is the Hellinger-SVD init pushed off its near-duplicates.
    Selection then takes the last healthy iterate (the barrier converges; median cost is not what
    is being optimized).
    """
    collision_only = cfg.mode == 'svd_collision'
    dev = E0.device
    E = E0.clone().float().requires_grad_(True)
    opt = torch.optim.Adam([E], lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), eps=1e-8)
    steps = cfg.optim.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(cfg.optim.warmup, 1))
        * (cfg.optim.min_lr_frac + (1 - cfg.optim.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))),
    )
    pi_f = tab.pi.float()
    w_base = tab.pi[univ].float() ** cfg.force.pi_pow
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 3)
    sigmas = list(cfg.force.sigmas)
    c0, d_att, lam_att = cfg.force.c0, cfg.force.d_att, cfg.force.lam_att

    if not collision_only:
        idx, _ = knn_pairs(E.detach(), univ, cfg.force.knn)
        ia = torch.arange(len(univ), device=dev).repeat_interleave(cfg.force.knn)
        ib = idx.reshape(-1)
        cost = pair_cost_cached(cache, ia, ib)
    nn_idx, dmin = refresh_nn(E.detach(), cfg.collision.nn_k)
    best = {'median_cost': float('inf'), 'step': -1, 'E': None, 'metrics': None}
    m0 = nn_audit(E0, ev['cache'], ev['univ'])
    a_end0, _ = endpoint_alpha(
        E0.float(), ev['logpi'], (E0.float() ** 2).sum(1), ev['end_a'], ev['end_eps'],
        ev['ha'], cfg.eval.eps_H, cfg.eval.eps_err, cfg.eval.alpha_cap, 4096,
    )
    if dmin >= 0.5 * cfg.collision.delta0 and math.isfinite(a_end0):
        best.update(
            {'median_cost': m0['median_cost'], 'step': 0, 'E': E0.cpu().clone(), 'metrics': {**m0, 'alpha_end': round(a_end0, 2)}}
        )
    t0 = time.time()
    for step in range(1, steps + 1):
        if collision_only:
            loss_main = torch.zeros((), device=dev)
        else:
            sel = torch.randint(len(ia), (cfg.optim.pair_batch,), device=dev, generator=gen)
            a_g, b_g = univ[ia[sel]], univ[ib[sel]]
            c = cost[sel]
            w = w_base[ia[sel]]
            d2 = (E[a_g] - E[b_g]).square().sum(1)
            rep = torch.zeros_like(d2)
            for s in sigmas:
                rep = rep + torch.exp(-d2 / (2 * s * s))
            att = torch.relu(c0 - c) / c0 * torch.relu(d2.clamp_min(1e-12).sqrt() - d_att).square()
            loss_main = (w * (rep * torch.relu(c - c0) + lam_att * att)).sum() / w.sum()
        col, _ = barrier_pens(E, nn_idx, cfg.collision.delta0, 1e9)
        loss = loss_main + cfg.collision.lam * col
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([E], cfg.optim.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            project_(E, pi_f, cfg.power)
        if step % cfg.force.refresh_every == 0:
            if not collision_only:
                idx, _ = knn_pairs(E.detach(), univ, cfg.force.knn)
                ib = idx.reshape(-1)
                cost = pair_cost_cached(cache, ia, ib)
            nn_idx, dmin = refresh_nn(E.detach(), cfg.collision.nn_k)
        if step % cfg.eval.every == 0 or step == steps:
            m = nn_audit(E.detach(), ev['cache'], ev['univ'])
            a_end, (ent_end, err_end) = endpoint_alpha(
                E.detach(), ev['logpi'], (E.detach() ** 2).sum(1), ev['end_a'], ev['end_eps'],
                ev['ha'], cfg.eval.eps_H, cfg.eval.eps_err, cfg.eval.alpha_cap, 4096,
            )
            healthy = (dmin >= 0.5 * cfg.collision.delta0) and math.isfinite(a_end)
            m.update(
                {
                    'alpha_end': round(a_end, 2) if math.isfinite(a_end) else float('inf'),
                    'err_end': round(err_end, 5),
                    'ent_end': round(ent_end, 6),
                    'dmin': round(dmin, 4),
                    'healthy': bool(healthy),
                    'loss': round(float(loss_main.detach()), 5),
                    'collision': round(float(col.detach()), 6),
                    'sps': round(step / (time.time() - t0), 2),
                }
            )
            print(f'[eval {step}] {m}', flush=True)
            log({'step': step, **m})
            if healthy and (collision_only or m['median_cost'] < best['median_cost']):
                best.update(
                    {'median_cost': m['median_cost'], 'step': step, 'E': E.detach().cpu().clone(), 'metrics': m}
                )
    if best['E'] is None:
        m = nn_audit(E.detach(), ev['cache'], ev['univ'])
        best.update({'median_cost': m['median_cost'], 'step': -1, 'E': E.detach().cpu().clone(), 'metrics': m})
    torch.save({'E': best['E'], 'step': best['step'], 'metrics': best['metrics']}, os.path.join(run_dir, 'E_best.pt'))
    return best


# Self-test


def self_test():
    """Check the sparse/structured estimators against dense references on a synthetic corpus."""
    torch.manual_seed(0)
    dt = torch.float64
    V, d, L = 24, 3, 64
    gen = torch.Generator().manual_seed(1)
    # Structured synthetic corpus: next token depends on the previous token's class
    tok = torch.zeros(400, L, dtype=torch.long)
    for w in range(400):
        s = int(torch.randint(V, (1,), generator=gen))
        for i in range(L):
            tok[w, i] = s
            s = ((s % 6) * 4 + int(torch.randint(4, (1,), generator=gen))) % V

    # Both-neighbor stats: merge invariants, I(A;C) sparse/closed form vs dense double sum
    keys, cnts = count_both_neighbors([tok[:200], tok[200:]], V)
    k1, c1 = count_both_neighbors([tok], V)
    assert torch.equal(keys, k1) and torch.equal(cnts, c1), 'chunked merge != one-shot count'
    stats = ContextStats(keys, cnts, V, 1, qc_floor=1e-2)
    tau = 7.0
    N = torch.zeros(V, stats.C_dim, dtype=dt)
    N[stats.row.long(), stats.col.long()] = stats.cnt.double()
    Q = (N + tau * stats.qc[None, :]) / (stats.Na[:, None] + tau)
    assert torch.allclose(Q.sum(1), torch.ones(V, dtype=dt), atol=1e-12), 'Q rows must sum to 1'
    iac_dense = float((stats.pi[:, None] * Q * (Q / stats.qc[None, :]).log()).sum())
    assert abs(stats.iac(tau) - iac_dense) < 1e-9, (stats.iac(tau), iac_dense)
    print('PASS counts / I(A;C) vs dense')

    # Hellinger SVD: structured matvec + init vs dense reference
    H = Q.sqrt()
    hbar = stats.pi @ H
    Bm = stats.pi.sqrt()[:, None] * (H - hbar[None, :])
    E0 = hellinger_svd_init(stats, tau, d, 'cpu', gen, oversample=12, n_iter=3)
    _, _, Vh = torch.linalg.svd(Bm.float(), full_matrices=False)
    E0_dense = (H.float() - hbar.float()[None, :]) @ Vh[:d].T
    # Compare column spans (rotations/signs are free): principal angles ~ 1
    Qa, _ = torch.linalg.qr(E0)
    Qb, _ = torch.linalg.qr(E0_dense)
    sv = torch.linalg.svdvals(Qa.T @ Qb)
    assert float(sv.min()) > 0.999, f'SVD init span mismatch: {sv}'
    print('PASS structured Hellinger-SVD init == dense reference (span)')

    # Adjacent-pair tables: dense rows/cols vs dense reference
    keys, cnts = count_adjacent([tok[:200]], V)
    tab = PairTable(keys, cnts, V)
    tabT = tab.transpose()
    tau = 3.0
    N = torch.zeros(V, V, dtype=dt)
    N[tab.row, tab.col] = tab.cnt
    Q = (N + tau * tab.q0[None, :]) / (tab.Na[:, None] + tau)
    a = torch.arange(V)
    assert torch.allclose(tab.dense_rows(a, tau), Q, atol=1e-12), 'dense_rows mismatch'
    got = tab.dense_cols(a, tab.csc_struct(), tau)
    assert torch.allclose(got.double(), Q.T, atol=1e-5), 'dense_cols mismatch'
    print('PASS PairTable dense rows/cols vs dense reference')

    # Cached pair cost vs direct fp64 formula
    cache = build_cost_cache(tab, tabT, a, tau, chunk=8)
    ia = torch.tensor([0, 3, 7, 11])
    ib = torch.tensor([5, 2, 19, 4])
    got = pair_cost_cached(cache, ia, ib, chunk=2)
    QT = (N.T + tau * tabT.q0[None, :]) / (tabT.Na[:, None] + tau)
    for j in range(len(ia)):
        aa, bb = int(ia[j]), int(ib[j])
        costL = float((QT[aa] * (Q[:, aa].clamp_min(1e-38).log() - Q[:, bb].clamp_min(1e-38).log())).sum())
        costR = float((Q[aa] * (QT[:, aa].clamp_min(1e-38).log() - QT[:, bb].clamp_min(1e-38).log())).sum())
        ref = costL + costR
        assert abs(float(got[j]) - ref) < 0.05 * max(1.0, abs(ref)), (float(got[j]), ref)
    print('PASS cached commit cost vs dense fp64 reference (bf16 tolerance)')

    # Force signs: costly pair -> repulsive gradient (d loss / d d2 < 0 means pushing apart
    # reduces loss); substitutable far pair -> attractive
    d2 = torch.tensor([0.09], requires_grad=True)
    rep = sum(torch.exp(-d2 / (2 * s * s)) for s in [0.15, 0.3, 0.6])
    (rep * 3.0).sum().backward()
    assert float(d2.grad) < 0, 'costly pair must be repelled'
    dd = torch.tensor([1.0], requires_grad=True)
    torch.relu(dd - 0.3).square().sum().backward()
    assert float(dd.grad) > 0, 'substitutable far pair must be attracted'
    print('PASS force-field signs')

    # Projection gauge
    pi_f = stats.pi
    Ec = torch.randn(V, d, dtype=dt, generator=gen) + 3.0
    project_(Ec, pi_f, float(d))
    assert float((pi_f @ Ec).norm()) < 1e-10
    assert abs(float((pi_f * Ec.square().sum(1)).sum()) - d) < 1e-9
    print('PASS pi-weighted projection gauge')


# Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-name', help='embedding config in configs/ (required unless --self-test)')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='tiny debugging run (throwaway sizes)')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.config_name:
        ap.error('--config-name is required')

    cfg = OmegaConf.load(os.path.join(REPO_ROOT, 'configs', f'{args.config_name}.yaml'))
    if args.smoke:
        cfg.data.stats_windows, cfg.data.stats_chunks, cfg.data.val_windows = 65536, 8, 8192
        cfg.force.universe, cfg.force.knn = 4096, 8
        cfg.optim.steps, cfg.eval.every, cfg.force.refresh_every = 300, 100, 50
        cfg.run_name = f'{cfg.run_name}_smoke'

    t0 = time.time()
    torch.manual_seed(cfg.seed)
    assert torch.cuda.is_available()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    cpu_gen = torch.Generator().manual_seed(cfg.seed + 1)
    gpu_gen = torch.Generator(device=dev).manual_seed(cfg.seed + 2)
    V, d = cfg.vocab_size, cfg.flow_dim

    job = os.environ.get('SLURM_JOB_ID') or os.environ.get('TORCHELASTIC_RUN_ID')
    job = job or datetime.datetime.now().strftime('%H%M%S')
    run_dir = os.path.join(REPO_ROOT, cfg.out_root, f'{cfg.run_name}_{datetime.datetime.now():%Y%m%d}_{job}')
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'resolved_config.yaml'))
    log_f = open(os.path.join(run_dir, 'log.jsonl'), 'a')

    def log(rec):
        rec['t'] = round(time.time() - t0, 1)
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()

    print(f'[run] dir={run_dir} device={torch.cuda.get_device_name()}', flush=True)

    # Counts: SHAPE / EVAL slices + R=1 stats for the SVD init
    train_ds = load_owt_split('openwebtext-train', 'train', cfg.data.cache_dir, cfg.data.block_size)
    valid_ds = load_owt_split('openwebtext-valid', 'validation', cfg.data.cache_dir, cfg.data.block_size)
    shape_chunks, eval_chunks = [], []
    n_oov = 0
    for i, rows in enumerate(strided_chunks(train_ds, cfg.data.stats_windows, cfg.data.stats_chunks, cpu_gen)):
        tok = rows.to(dev, non_blocking=True).long()
        n_oov += int((tok >= V).sum())  # Stray [PAD] windows in the cache; pairs are filtered
        (eval_chunks if i % 4 == 3 else shape_chunks).append(tok)
    if n_oov:
        print(f'[counts] dropping pairs touching {n_oov} out-of-vocab tokens', flush=True)
    n_shape = sum(t.numel() for t in shape_chunks)
    n_eval = sum(t.numel() for t in eval_chunks)
    print(f'[counts] shape={n_shape / 1e6:.0f}M tokens, eval={n_eval / 1e6:.0f}M tokens', flush=True)

    keys, cnts = count_adjacent(shape_chunks, V)
    tab = PairTable(keys, cnts, V)
    tabT = tab.transpose()
    keysE, cntsE = count_adjacent(eval_chunks, V)
    tabE = PairTable(keysE, cntsE, V)
    tabET = tabE.transpose()

    nv = min(cfg.data.val_windows, len(valid_ds))
    val_tok = valid_ds[:nv]['input_ids'].to(dev).long()
    val_tok = val_tok[(val_tok < V).all(dim=1)]  # Drop rare stray-[PAD] windows
    va = val_tok[:, :-1].reshape(-1)
    vb = val_tok[:, 1:].reshape(-1)
    sel = torch.randperm(len(va), device=dev, generator=gpu_gen)[: cfg.tau.n_pairs]
    tau, tau_losses = tab.select_tau(va[sel], vb[sel], list(cfg.tau.grid))
    print(f'[tau] bigram tau={tau:g} (heldout logloss {tau_losses[tau]:.4f})', flush=True)
    log({'phase': 'tau', 'tau': tau, 'losses': {str(k): round(v, 5) for k, v in tau_losses.items()}})

    # R=1 both-neighbor `ContextStats` (probe layout off_idx * V + ctx) for the Hellinger-SVD init
    R = 1
    keys1, cnts1 = count_both_neighbors(shape_chunks, V)
    stats = ContextStats(keys1, cnts1, V, R, cfg.counts.qc_floor)
    pa, pc = sample_probe_pairs(val_tok, cfg.tau.n_pairs, R, V, gpu_gen)
    tau1, _ = stats.select_tau(pa, pc, list(cfg.tau.grid))
    del val_tok, keys1, cnts1, shape_chunks, eval_chunks
    torch.cuda.empty_cache()
    print(f'[svd] R=1 probe tau={tau1:g}, I(A;C)={stats.iac(tau1):.4f} nats', flush=True)

    E0 = hellinger_svd_init(stats, tau1, d, dev, gpu_gen)
    n_dup = dedup_jitter_(E0, cfg.collision.delta0, gpu_gen)
    project_(E0, tab.pi.float(), cfg.power)
    print(f'[svd-init] jittered {n_dup} duplicate rows', flush=True)

    # Caches, eval structures.
    # Only 'force_field' needs the SHAPE cost cache; the EVAL cache is built for the held-out NN
    # audit that every asset records.
    mode = cfg.mode
    if mode not in ('force_field', 'hellinger_svd', 'svd_collision'):
        raise ValueError(f'unknown mode {mode!r}')
    univ = tab.pi.argsort(descending=True)[: cfg.force.universe]
    cache = None
    if mode == 'force_field':
        print('[cache] building SHAPE cost cache', flush=True)
        cache = build_cost_cache(tab, tabT, univ, tau)
    eval_univ = tabE.pi.argsort(descending=True)[: cfg.eval.audit_top]
    print('[cache] building EVAL cost cache', flush=True)
    eval_cache = build_cost_cache(tabE, tabET, eval_univ, tau)
    logpi = torch.where(tab.pi > 0, tab.pi.log(), torch.tensor(-1e30, dtype=torch.float64, device=dev)).float()
    cum = tab.pi.cumsum(0)
    u = torch.rand(cfg.eval.endpoint_n, dtype=torch.float64, device=dev, generator=gpu_gen) * cum[-1]
    end_a = torch.searchsorted(cum, u).clamp_max_(V - 1)
    p = tab.pi[tab.pi > 0]
    ev = {
        'cache': eval_cache,
        'univ': eval_univ,
        'logpi': logpi,
        'ha': float(-(p * p.log()).sum()),
        'end_a': end_a,
        'end_eps': torch.randn(cfg.eval.endpoint_n, d, device=dev, generator=gpu_gen),
    }
    m0 = nn_audit(E0, eval_cache, eval_univ)
    print(f'[init audit] {m0}', flush=True)
    log({'step': 0, **m0})

    # Force field + selection + export
    if mode == 'hellinger_svd':
        _, dmin0 = refresh_nn(E0, cfg.collision.nn_k)
        a_end0, (ent0, err0) = endpoint_alpha(
            E0.float(), ev['logpi'], (E0.float() ** 2).sum(1), ev['end_a'], ev['end_eps'],
            ev['ha'], cfg.eval.eps_H, cfg.eval.eps_err, cfg.eval.alpha_cap, 4096,
        )
        metrics0 = {
            **m0,
            'alpha_end': round(a_end0, 2) if math.isfinite(a_end0) else float('inf'),
            'err_end': round(err0, 5),
            'ent_end': round(ent0, 6),
            'dmin': round(dmin0, 4),
            'healthy': bool(dmin0 >= 0.5 * cfg.collision.delta0 and math.isfinite(a_end0)),
        }
        print(f'[svd-init export] {metrics0}', flush=True)
        best = {'median_cost': m0['median_cost'], 'step': 0, 'E': E0.cpu().clone(), 'metrics': metrics0}
        torch.save({'E': best['E'], 'step': 0, 'metrics': metrics0}, os.path.join(run_dir, 'E_best.pt'))
    else:
        best = train_force_field(E0, tab, cache, univ, ev, cfg, log, run_dir)
    E_win = best['E'].to(dev).float()
    perms = []
    for _ in range(2):
        perm = torch.randperm(V, generator=cpu_gen).to(dev)
        perms.append(nn_audit(E_win[perm], eval_cache, eval_univ)['median_cost'])
    m_final = best['metrics']
    print(f'[control] median_cost={m_final["median_cost"]} permuted={perms}', flush=True)
    log({'phase': 'final', 'metrics': m_final, 'perm': perms})

    emb = torch.cat([E_win.cpu(), torch.zeros(1, d)], dim=0)
    source = {
        'force_field': 'Hellinger-SVD init + commit-cost force field',
        'hellinger_svd': 'R=1 both-neighbor Hellinger-SVD init only (no optimizer steps)',
        'svd_collision': 'Hellinger-SVD init + collision barrier steps only (no commit-cost force field)',
    }[mode]
    out_name = cfg.out_name
    torch.save(
        {
            'embedding': emb,
            'source': f'{source}, mode={mode}',
            'mode': mode,
            'dim': d,
            'gauge': f'pi-weighted zero mean, pi-weighted power P={float(cfg.power):g}; do NOT rescale '
            '(any paired time schedule must be computed at exactly this scale)',
            'pi': tab.pi.cpu().float(),
            'tau_bigram': float(tau),
            'tau_probe': float(tau1),
            'universe': int(cfg.force.universe),
            'stats_tokens': {'shape': n_shape, 'eval': n_eval},
            'selected': {'step': best['step'], 'metrics': m_final},
            'perm_control_median_cost': perms,
            'pad_row': 'zeros appended for [PAD] id 50257',
            'config': OmegaConf.to_yaml(cfg),
        },
        os.path.join(run_dir, out_name),
    )
    summary = {
        'final': m_final,
        'init_audit': m0,
        'perm_control': perms,
        'tau_bigram': float(tau),
        'tau_probe': float(tau1),
        'run_dir': run_dir,
        'total_min': round((time.time() - t0) / 60, 1),
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'[done] median_cost={m_final["median_cost"]} (perm {perms}); assets in {run_dir}', flush=True)
    log_f.close()


if __name__ == '__main__':
    main()
