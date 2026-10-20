"""Inference samplers of the ReLF (`SamplerMixin`) and the deterministic grid / gate helpers.

Mixed into `algo.ReLF`; the methods read the model state (`_backbone_forward`, `token_embed`, `logit_mask`,
`prior_mean`, the asset schedule) and the settings `_init_sampler` parses.
"""

import torch

from relf import utils
from relf.fused_ce import CE_CHUNK_ROWS, _add_singular_logits

GRID_FAMILIES = ('uniform', 'arc_length', 'knots')


def grid_family_map(fam, u, knots=None):
    """Map u -> t(u) on [0, 1] for the 'knots' grid family.

    The monotone piecewise-linear map through the knots (0, 0), (u1, v1), ..., (uK, vK), (1, 1),
    with `knots` = (u1, v1, ..., uK, vK) strictly increasing in both coordinates inside (0, 1); u
    strictly inside (0, 1), fp64. Pure function behind
    `ReLF.grid_nodes`. A knot with v = 0.999999 collapses the following nodes onto t = b
    (the stationary self-conditioning iterations of the frozen schedule).
    """
    if fam != 'knots':
        raise ValueError(f'Unknown sampling.grid: {fam}')
    x = torch.tensor([0.0] + [float(k) for k in knots[0::2]] + [1.0], dtype=torch.float64, device=u.device)
    y = torch.tensor([0.0] + [float(k) for k in knots[1::2]] + [1.0], dtype=torch.float64, device=u.device)
    h = x[1:] - x[:-1]
    d = (y[1:] - y[:-1]) / h  # Secant slopes, all > 0 for monotone knots
    idx = (torch.searchsorted(x, u, right=True) - 1).clamp(0, len(h) - 1)
    return y[idx] + d[idx] * (u - x[idx])


def validate_knots(fam, knots):
    """Check knots (u1, v1, ..., uK, vK): both coordinates strictly increasing inside (0, 1)."""
    if len(knots) < 2 or len(knots) % 2:
        raise ValueError(f'sampling.grid {fam}: knots must be a flat list u1, v1, ..., uK, vK, got {knots}')
    for seq in (knots[0::2], knots[1::2]):
        if not all(a < b for a, b in zip((0.0,) + tuple(seq), tuple(seq) + (1.0,))):
            raise ValueError(f'sampling.grid {fam} needs strictly increasing knots inside (0, 1), got {knots}')


def parse_gate(name, spec):
    """Parse a piecewise-linear gate g(t) from a flat knot list [t1, g1, ..., tK, gK].

    t strictly increasing in (0, 1), g >= 0. Returns a tuple of (t, g) pairs, or None (g == 1) for a
    None `spec`.
    """
    if spec is None:
        return None
    vals = [float(v) for v in spec]
    if len(vals) < 2 or len(vals) % 2:
        raise ValueError(f'sampling.{name} must be a flat list t1, g1, ..., tK, gK')
    ts, gs = vals[0::2], vals[1::2]
    if not all(0.0 < a < b < 1.0 for a, b in zip(ts[:-1], ts[1:])) or not (0.0 < ts[0] < 1.0 and ts[-1] < 1.0) or min(gs) < 0.0:
        raise ValueError(f'sampling.{name}: knots need strictly increasing t in (0, 1) and g >= 0, got {vals}')
    return tuple(zip(ts, gs))


def eval_gate(gate, t):
    """Evaluate g(t) for a `parse_gate` tuple: linear between knots, constant outside them."""
    if gate is None:
        return 1.0
    if t <= gate[0][0]:
        return gate[0][1]
    if t >= gate[-1][0]:
        return gate[-1][1]
    for (t0, g0), (t1, g1) in zip(gate[:-1], gate[1:]):
        if t0 <= t <= t1:
            return g0 + (g1 - g0) * (t - t0) / (t1 - t0)
    raise AssertionError('unreachable')


def fixed_endpoint_nodes(fam, N, a, b, knots=None, device=None):
    """Return the 'knots' forward nodes with pinned first / last sampling times.

    t_0 = a and t_{N-1} = b are assigned exactly; the interior is
    redistributed by the knot map on the normalized coordinate u_i = i / (N - 1):
    t_i = a + (b - a) F(u_i), fp64.
    """
    N = int(N)
    if N < 2:
        raise ValueError('fixed-endpoint grids need num_steps >= 2')
    if not (0.0 < a < b < 1.0):
        raise ValueError(f'fixed endpoints need 0 < a < b < 1, got ({a}, {b})')
    u = torch.arange(N, dtype=torch.float64, device=device) / (N - 1)
    F = grid_family_map(fam, u, knots=knots)
    t = a + (b - a) * F
    t[0], t[-1] = a, b
    if not bool(torch.all(t[1:] > t[:-1])):
        raise ValueError(f'fixed-endpoint {fam} grid is not strictly increasing')
    return t


class SamplerMixin:
    """`generate_samples` and the euler / sde / heun / mh samplers of `algo.ReLF`."""

    def _init_sampler(self, config):
        """Read the sampler settings: `sampling.grid` and the `eval_sampler` section."""
        es = config.eval_sampler
        self.sampling_grid = utils.config_choice(config.sampling, 'grid', GRID_FAMILIES)
        self.sampling_solver = utils.config_choice(es, 'solver', ('euler', 'heun', 'sde'))
        self.carry_mode = utils.config_choice(es, 'carry_mode', ('warm', 'cold'))
        self.sde_noise_scale = float(es.sde_noise_scale)
        self.sde_noise_decay = float(es.sde_noise_decay)
        # Momentum guidance (arXiv 2602.20360): velocity (1 + alpha) v - alpha m, m = EMA_beta(v)
        self.momentum_guidance = bool(es.momentum_guidance)
        self.mg_alpha = float(es.mg_alpha)
        self.mg_beta = float(es.mg_beta)
        self.x1_extrap = float(es.x1_extrap)
        self.grid_midpoint = bool(es.grid_midpoint)
        # 'knots': knot map between the pinned endpoints t_0 = grid_a, t_{N-1} = grid_b
        self.grid_a = self.grid_b = self.grid_knots = None
        if self.sampling_grid == 'knots':
            self.grid_a, self.grid_b = float(es.grid_a), float(es.grid_b)
            self.grid_knots = tuple(float(k) for k in es.grid_knots)
            validate_knots('knots', self.grid_knots)
        # 'mh' state form: z = t m + (1 - t) eta with gated sde refresh (`mh_plan`)
        self.state_form = utils.config_choice(es, 'state_form', ('z', 'mh'))
        self.sde_eta_gate = parse_gate('sde_eta_gate', es.sde_eta_gate)
        self.sde_mix_gate = parse_gate('sde_mix_gate', es.sde_mix_gate)
        self.sde_gate_normalize = bool(es.sde_gate_normalize)
        self.decode_t = None if es.decode_t is None else float(es.decode_t)
        self.eta_strat = utils.config_choice(es, 'eta_strat', ('off', 'std'))

    @torch.no_grad()
    def _init_state(self, num_samples, device, z0=None):
        """Return the sampler's source point: a fresh N(0, I) draw or the supplied `z0` in fp32.

        The fresh draw keeps the historical call (byte-identical RNG consumption). `z0`
        ((B, L, flow_dim), any float dtype) is the coupling hook of the reflow pair collection
        (reflow.py). It is the t = 0 state of the 'z' paths and the initial eta of the 'mh' path
        (z = t m + (1 - t) eta), so in both forms x_t = t x1 + (1 - t) z0 is the line the pair
        trains on. A supplied z0 is used verbatim -- the mh path's `eta_strat` standardization is
        NOT re-applied (the collector applies it before storing).
        """
        shape = (num_samples, self.num_tokens, self.flow_dim)
        if z0 is None:
            return torch.randn(shape, device=device, dtype=torch.float32)
        if tuple(z0.shape) != shape:
            raise ValueError(f'z0 shape {tuple(z0.shape)} != {shape}')
        return z0.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, z0=None):
        """Generate samples, dispatching on `sampling.state_form` and `sampling.solver`.

        `state_form: mh` -> `_generate_samples_mh`, `solver: heun` -> `_generate_samples_heun`,
        otherwise euler / sde on any `sampling.grid` via `_generate_samples_general`.

        `z0`: optional supplied source point (see `_init_state`); None draws N(0, I).

        The ODE state, time grid, softmax expectation, and Euler update are pinned to fp32
        regardless of the model dtype: a bf16 integrator state loses updates below the ULP of `z`
        (sample collapse at high step counts) and mimics a temperature drop at moderate ones (cf. the MDM
        analogue, arXiv:2409.02908). Only the network forward runs in the
        model's own precision.

        TF32 must be off (main.py sets it process-wide; callers outside main.py set it themselves):
        at 10-bit mantissa the sampler's fp32 matmuls are the only above-chaos-floor error source
        besides discretization (0.7-1.7% token flips).
        """
        if num_steps is None:
            num_steps = self.relf_config.sampling.steps
        if self.state_form == 'mh':
            return self._generate_samples_mh(num_samples, num_steps, z0=z0)
        if self.sampling_solver == 'heun':
            return self._generate_samples_heun(num_samples, num_steps, z0=z0)
        return self._generate_samples_general(num_samples, num_steps, z0=z0)

    def _x1_or_ids(self, z, t_in, token_embed, last, x_sc=None):
        """Run one model forward at time `t_in` with row-chunked logit post-processing.

        Post-processing is the singular addmm and the padded-slot mask. Returns argmax ids (B, L)
        when `last`, else the fp32 softmax expectation x1_hat (B, L, flow_dim).
        """
        B, L = z.shape[0], z.shape[1]
        device = z.device
        logits = self._backbone_forward(z.to(self.dtype), t_in, x_sc=x_sc)
        flat = logits.reshape(-1, logits.shape[-1])
        n_rows = flat.shape[0]
        sing = self._singular_desc(z, t_in) if self.singular_logits else None
        if last:
            ids = torch.empty(n_rows, device=device, dtype=torch.long)
        if not last:
            x_1_embed = torch.empty((n_rows, self.flow_dim), device=device, dtype=torch.float32)
        for s in range(0, n_rows, CE_CHUNK_ROWS):
            e = min(s + CE_CHUNK_ROWS, n_rows)
            chunk = flat[s:e].float()
            if sing is not None:
                _add_singular_logits(chunk, sing, s, e)
            chunk.add_(self.logit_mask)
            if last:
                ids[s:e] = chunk.argmax(dim=-1)
            if not last:
                x_1_embed[s:e] = self._softmax_x1(chunk, token_embed)
        del logits, flat
        if last:
            return ids.view(B, L)
        return x_1_embed.view(B, L, self.flow_dim)

    def _generate_samples_heun(self, num_samples, num_steps, z0=None):
        """Sample with Heun (2nd order, 2 NFE/step) on the uniform grid, in the x1 parametrization.

        Shared frame with the euler path: fp32 state, forwards at t_i = i / N for i in 0..N-1,
        argmax decode at the last interior node t = (N-1)/N, velocity v = (x1_hat - z) * N / (N - i)
        with the exact rational 1 - t_i = (N - i) / N. Predict with v at t_i,
        correct with the average of v(t_i) and v(t_{i+1}) at the predicted state. The
        self-conditioning carry updates right after each network call, so the corrector conditions
        on the predictor's estimate.
        """
        N = num_steps
        if N < 1:
            raise ValueError(f'num_steps must be >= 1, got {N}')
        B = num_samples
        L = self.num_tokens
        device = self.device

        token_embed = self._embed_matrix()
        t_vals = torch.linspace(0.0, 1.0, N + 1, device=device, dtype=torch.float32)
        z = self._init_state(B, device, z0)
        x_sc = (
            torch.zeros((B, L, self.flow_dim), device=device, dtype=torch.float32)
            if self.self_cond
            else None
        )

        for i in range(N):
            t_in = t_vals[i].expand(B)
            if i == N - 1:
                return self._x1_or_ids(z, t_in, token_embed, last=True, x_sc=x_sc)
            x1 = self._x1_or_ids(z, t_in, token_embed, last=False, x_sc=x_sc)
            if self.self_cond and self.carry_mode == 'warm':
                x_sc = x1
            # Exact-rational velocity: v = (x1 - z) / (1 - t_i), 1 - t_i = (N - i)/N
            v = (x1 - z) * (N / (N - i))
            z_pred = z + v / N
            t_next_in = t_vals[i + 1].expand(B)
            x1_next = self._x1_or_ids(z_pred, t_next_in, token_embed, last=False, x_sc=x_sc)
            if self.self_cond and self.carry_mode == 'warm':
                x_sc = x1_next
            v_next = (x1_next - z_pred) * (N / (N - i - 1))
            z = z + (v + v_next) / (2.0 * N)

        raise ValueError(f'num_steps must be >= 1, got {num_steps}')

    # General deterministic time grids
    def grid_nodes(self, num_steps, device=None):
        """Return the forward nodes 0 <= t_0 < t_1 < ... < t_{N-1} (fp64) for `sampling.grid`.

        N forwards, N-1 updates, argmax decode at t_{N-1}; deterministic. Legacy placement
        (`grid_midpoint` False): 'uniform' keeps the nodes i/N (decode at (N-1)/N); 'arc_length' uses the
        asset schedule t(u_k), u_k = k/(N-1) (decode at t_end). Midpoint placement (`grid_midpoint`
        True): tau_i = (i + 0.5) / N through the same t(tau) ('uniform' -> tau itself, 'arc_length' -> the
        table), so no node sits at 0 or t_end. 'knots' is always the fixed-endpoint knot map
        t_0 = grid_a, t_{N-1} = grid_b (`fixed_endpoint_nodes`).
        """
        N = int(num_steps)
        if N < 1:
            raise ValueError(f'num_steps must be >= 1, got {N}')
        device = device or self.device
        fam = self.sampling_grid
        f64 = dict(device=device, dtype=torch.float64)
        if fam == 'knots':
            return fixed_endpoint_nodes('knots', N, self.grid_a, self.grid_b, knots=self.grid_knots, device=device)
        if self.grid_midpoint:
            tau = (torch.arange(N, **f64) + 0.5) / N
            if fam == 'uniform':
                return tau
            alpha = self._lut_alpha(tau)
            return alpha / (1.0 + alpha)
        if fam == 'uniform':
            return torch.arange(N, **f64) / N
        if N == 1:
            return torch.zeros(1, **f64)
        u = torch.arange(N, **f64) / (N - 1)
        alpha = self._lut_alpha(u)
        return alpha / (1.0 + alpha)

    def _generate_samples_general(self, num_samples, num_steps, z0=None):
        """Sample with euler / sde (+ momentum guidance) on any grid from `grid_nodes`.

        Per interval i (0 <= i < N-1): dt_i = t_{i+1} - t_i, beta_t = 1 - t_i, velocity
        v = (x1_hat - z) / (1 - t_i), MG decay beta_i = beta^(N dt_i) (= beta on the uniform grid),
        sde refresh beta_noised = min(dt_i * noise * beta_t^decay, beta_t) and the curved update
        z = z_noised + (x1_eff - z_noised) * dt_i / (1 - t_i). Scalars are fp64 Python floats
        pre-rounded to fp32 by the tensor ops; the state, expectation and updates stay fp32. On the
        'uniform' family the scalars are exact rationals (N - i, N / (N - i), i / (N - i), 1 / N),
        never floating-point 1 - t.
        """
        N = int(num_steps)
        if N < 1:
            raise ValueError(f'num_steps must be >= 1, got {N}')
        solver = self.sampling_solver
        if solver not in ('euler', 'sde'):
            raise ValueError(f'general-grid sampling supports euler and sde only, got {solver}')
        B = num_samples
        L = self.num_tokens
        device = self.device
        uniform = self.sampling_grid == 'uniform' and not self.grid_midpoint
        nodes = self.grid_nodes(N, device)
        # Forward-time inputs: fp32 linspace on the uniform family, else the fp64 nodes rounded once
        t_vals = (torch.linspace(0.0, 1.0, N + 1, device=device, dtype=torch.float32)[:N]
                  if uniform else nodes.float())
        nodes_l = nodes.tolist()

        token_embed = self._embed_matrix()
        z = self._init_state(B, device, z0)
        t0 = nodes_l[0]
        if t0 > 0.0:
            # First node inside (0, 1) (midpoint grids): start at the moment-matched marginal
            # (1 - t_0) x0 + t_0 m, m = the asset's pi-weighted embedding mean
            z = z * (1.0 - t0) + t0 * self.prior_mean.to(device=device, dtype=torch.float32)
        mg_m = None
        x_sc = (
            torch.zeros((B, L, self.flow_dim), device=device, dtype=torch.float32)
            if self.self_cond
            else None
        )
        x1_extrap = self.x1_extrap
        x1_prev, da_prev = None, None

        for i in range(N):
            t_in = t_vals[i].expand(B)
            last = i == N - 1
            if last:
                return self._x1_or_ids(z, t_in, token_embed, last=True, x_sc=x_sc)
            x1 = self._x1_or_ids(z, t_in, token_embed, last=False, x_sc=x_sc)
            if self.self_cond and self.carry_mode == 'warm':
                x_sc = x1
            # Interval scalars (fp64 Python floats; exact rationals on the uniform family)
            if uniform:
                dt = 1.0 / N
                omt = (N - i) / N  # 1 - t_i
                inv_omt = N / (N - i)  # 1 / (1 - t_i)
                t_over_omt = i / (N - i)  # t_i / (1 - t_i)
                denom = N - i  # (1 - t_i) / dt_i
                inv_dt = N  # 1 / dt_i
                n_dt = 1.0  # N * dt_i
            else:
                t_i, t_n = nodes_l[i], nodes_l[i + 1]
                dt = t_n - t_i
                omt = 1.0 - t_i
                inv_omt = 1.0 / omt
                t_over_omt = t_i / omt
                denom = omt / dt
                inv_dt = 1.0 / dt
                n_dt = N * dt
            v = (x1 - z) * inv_omt
            # Second-order x1 extrapolation in alpha = t / (1 - t) (AB2 / DPM++2M form):
            # the curved update is left-rectangle quadrature of
            # int x1_hat d alpha; x1_upd = x1 + c * da / (2 da_prev) * (x1 - x1_prev) removes the
            # first-order lag. Used ONLY in the deterministic update; the refresh (x0_pred) and the
            # self-conditioning carry stay on the raw x1. x1_extrap = 0 skips the block
            # (bit-identical).
            x1_upd = x1
            if x1_extrap:
                ti, tn = (i / N, (i + 1) / N) if uniform else (nodes_l[i], nodes_l[i + 1])
                da = (tn - ti) / ((1.0 - ti) * (1.0 - tn))
                if x1_prev is not None:
                    x1_upd = x1 + (x1 - x1_prev) * (x1_extrap * da / (2.0 * da_prev))
                x1_prev, da_prev = x1, da
            if self.momentum_guidance:
                if mg_m is None:
                    mg_m = v
                beta_i = self.mg_beta ** n_dt
                v_g = (1.0 + self.mg_alpha) * v - self.mg_alpha * mg_m
                mg_m = (1.0 - beta_i) * v + beta_i * mg_m
            if solver == 'euler':
                if not self.momentum_guidance:
                    z = z + (x1_upd - z) / denom
                else:
                    z = z + v_g / inv_dt
            else:  # sde
                x1_eff = x1_upd if not self.momentum_guidance else z + v_g * omt
                x0_pred = z * inv_omt - x1 * t_over_omt if not self.momentum_guidance else z * inv_omt - x1_eff * t_over_omt
                beta_t = omt
                beta_noised = min(max(dt * self.sde_noise_scale * (beta_t ** self.sde_noise_decay), 0.0), beta_t)
                noise_std = (beta_t ** 2 - (beta_t - beta_noised) ** 2) ** 0.5
                if beta_noised == 0.0:
                    z_noised = z
                else:
                    z_noised = z - beta_noised * x0_pred + noise_std * torch.randn_like(z)
                z = z_noised + (x1_eff - z_noised) / denom

        raise ValueError(f'num_steps must be >= 1, got {num_steps}')

    def mh_plan(self, num_steps, split=False):
        """Return the per-forward plan of the 'mh' sampler and its decode time.

        The plan is a list of (t_state, kap_eta, w) for the N-1 updates. In (m, eta) coordinates the
        plain sde step is m <- m + kap (x1 - m), eta <- (1 - kap) eta + sqrt(1 - (1 - kap)^2) eps
        with kap = dt nu (1 - t)^(decay - 1), followed by the curved ODE step m <- m + lam (x1 - m),
        lam = dt / (t_{i+1} (1 - t_i)) (= 1 - s_i / s_{i+1}); net mixing
        w = 1 - (1 - kap_m)(1 - lam). kap_eta and kap_m share nu = `sde_noise_scale` and carry their
        own gates (fp64 scalars); `split` returns (kap_m, lam) per interval instead.
        """
        N = int(num_steps)
        nodes_l = self.grid_nodes(N).tolist()
        sde = self.sampling_solver == 'sde'
        nu = float(self.sde_noise_scale) if sde else 0.0
        decay = float(self.sde_noise_decay) if sde else 1.0
        g_eta = [eval_gate(self.sde_eta_gate, t) for t in nodes_l[:-1]]
        g_m = [eval_gate(self.sde_mix_gate, t) for t in nodes_l[:-1]]
        if self.sde_gate_normalize and N > 1:
            # Mean-preserving: sum_i dt_i g_i == sum_i dt_i (the total refresh at constant scale is
            # unchanged)
            dts = [nodes_l[i + 1] - nodes_l[i] for i in range(N - 1)]
            for g in (g_eta, g_m):
                c = sum(d * gi for d, gi in zip(dts, g)) / sum(dts)
                if c > 0:
                    g[:] = [gi / c for gi in g]
        plan = []
        for i in range(N - 1):
            t, tn = nodes_l[i], nodes_l[i + 1]
            dt = tn - t
            lam = dt / (tn * (1.0 - t))
            kap_eta = min(dt * nu * g_eta[i] * (1.0 - t) ** (decay - 1.0), 1.0) if sde else 0.0
            kap_m = min(dt * nu * g_m[i] * (1.0 - t) ** (decay - 1.0), 1.0) if sde else 0.0
            w = 1.0 - (1.0 - kap_m) * (1.0 - lam)
            plan.append((kap_m, lam) if split else (t, kap_eta, w))
        t_dec = float(self.decode_t) if self.decode_t is not None else nodes_l[-1]
        return plan, t_dec

    def _generate_samples_mh(self, num_samples, num_steps, z0=None):
        """Sample with euler / sde (stable, curved) on any grid in (m, eta) coordinates.

        z_i = t_i m + (1 - t_i) eta, N forwards, argmax from the final forward at `decode_t`.
        With no gates and `decode_t` None this is the plain sde trajectory up
        to fp32 re-association. The trailing nodes a `knots` knot at
        v = 0.999999 collapses onto t = b have dt ~ 1e-7, hence kap ~ lam ~ 0: they re-run the
        network only to converge the self-conditioning carry the decode reads.
        """
        N = int(num_steps)
        B, L, device = num_samples, self.num_tokens, self.device
        plan, t_dec = self.mh_plan(N)
        token_embed = self._embed_matrix()
        strat = self.eta_strat == 'std'

        def _randn():
            # Gaussian draw; `eta_strat` 'std' centres and unit-scales it per flow dimension across
            # the positions (each row keeps its N(0, I) marginal; only the empirical spread of the L
            # rows changes)
            e = torch.randn((B, L, self.flow_dim), device=device, dtype=torch.float32)
            if strat:
                e = (e - e.mean(dim=1, keepdim=True)) / e.std(dim=1, keepdim=True).clamp_min(1e-6)
            return e

        # A supplied z0 is the initial eta verbatim (already standardized by the collector when
        # `eta_strat` is 'std'; reflow.py); None keeps the historical draw and RNG consumption
        eta = _randn() if z0 is None else self._init_state(B, device, z0)
        m = self.prior_mean.to(device=device, dtype=torch.float32).expand(B, L, self.flow_dim).contiguous()
        x_sc = torch.zeros((B, L, self.flow_dim), device=device, dtype=torch.float32) if self.self_cond else None
        x1_extrap = float(self.x1_extrap)
        x1_prev = None
        nodes_l = self.grid_nodes(N).tolist()
        if x1_extrap:
            # Per-interval d alpha; the extrapolation coefficient is c * da_i / (2 da_{i-1}), and
            # the net mixing w is split back into its sde (kap_m, raw x1) and ODE (lam, extrapolated
            # x1) factors
            da_list = [(nodes_l[i + 1] - nodes_l[i]) / ((1.0 - nodes_l[i]) * (1.0 - nodes_l[i + 1])) for i in range(N - 1)]
            plan_km = self.mh_plan(N, split=True)[0]
        # Momentum guidance on the mh path: the general
        # path's v-space transform v_g = (1 + a) v - a mg_m, mg_m <- (1 - beta_i) v + beta_i mg_m,
        # beta_i = beta^(N dt_i), rewritten on x1 so that a = 0 is bit-identical to no guidance:
        # x1_g = x1 + a (1 - t_i) (v - mg_m). Unlike the general path it stacks on the x1
        # extrapolation (v is taken from x1_upd); the same guidance shift a (1 - t_i)(v - mg_m) is
        # added to the raw x1 of the sde mixing and to the extrapolated x1 of the ODE step.
        mg_m = None
        for i in range(N):
            last = i == N - 1
            t = t_dec if last else plan[i][0]
            z = m * t + eta * (1.0 - t)
            t_in = torch.full((B,), float(t), device=device, dtype=torch.float32)
            if last:
                return self._x1_or_ids(z, t_in, token_embed, last=True, x_sc=x_sc)
            x1 = self._x1_or_ids(z, t_in, token_embed, last=False, x_sc=x_sc)
            # The extrapolated prediction is formed before the carry and the eta refresh (no RNG in
            # between)
            x1_upd = x1
            if x1_extrap:
                if x1_prev is not None:
                    x1_upd = x1 + (x1 - x1_prev) * (x1_extrap * da_list[i] / (2.0 * da_list[i - 1]))
                x1_prev = x1
            if self.self_cond and self.carry_mode == 'warm':
                x_sc = x1
            _, kap_eta, w = plan[i]
            if kap_eta > 0.0:
                rho = 1.0 - kap_eta
                eta = eta * rho + _randn() * ((1.0 - rho * rho) ** 0.5)
            if self.momentum_guidance:
                omt = 1.0 - float(t)
                v = (x1_upd - z) * (1.0 / omt)
                if mg_m is None:
                    mg_m = v
                beta_i = self.mg_beta ** (N * (nodes_l[i + 1] - nodes_l[i]))
                dx = (v - mg_m) * (self.mg_alpha * omt)
                mg_m = v * (1.0 - beta_i) + mg_m * beta_i
                x1_mix, x1_upd = x1 + dx, x1_upd + dx
            else:
                x1_mix = x1
            if x1_extrap:
                # The sde mixing (kap_m) uses the raw x1 and the ODE step (lam) the extrapolated
                # one
                kap_m, lam = plan_km[i]
                if kap_m > 0.0:
                    m = m + (x1_mix - m) * kap_m
                m = m + (x1_upd - m) * lam
            elif w > 0.0:
                m = m + (x1_mix - m) * w
        raise ValueError(f'num_steps must be >= 1, got {num_steps}')
