---
name: relf
description: Method reference for the Rectified Language Flow (ReLF) — flow space, corruption process, training objective, sampling ODE, the paper's Singular Logits parameterization, and the self-conditioning / distilled SC guidance extensions. Use when reasoning about, explaining, or modifying the ReLF method or its math.
---

# The Rectified Language Flow (ReLF) method

This skill is the detailed method document for this repo's single-time continuous Flow Language Model. Notation matches the code: the single time variable is `t`, `x0` is noise, `x1` is data, `xt` is the interpolant. The paper writes the sampling-time ODE state as `z_t` and positions as superscripts `w^(l)`; here we use `xt` / `z` and `w_l`, matching `algo.py`.

## Notation

| Symbol | Meaning | In code |
|---|---|---|
| `V` | real vocab size (50257 GPT-2 + `[PAD]` = 50258) | `ReLF.real_vocab_size`; head padded to `ReLF.vocab_size` (50304) |
| `L` | sequence length (1024) | `model.length` |
| `d` | flow-space dimension (32 for the headline hsvd32s asset) | `model.flow_dim` |
| `E` | fixed token embedding matrix, shape `(V, d)` | buffer `ReLF.token_embed` |
| `w_l` | token at position `l`; `w = (w_1..w_L)` | `input_ids` |
| `x1` | data point in flow space: `x1_l = E[w_l]` | `token_embed[x1]` |
| `x0` | noise, `x0_l ~ N(0, I_d)` i.i.d. per position | `torch.randn_like` |
| `t` | flow time in `[0, 1]`: `t=0` pure noise, `t=1` data | `t` |
| `xt` | interpolant `t * x1 + (1 - t) * x0` | `corrupt_continuous` |
| `f_theta(xt, t)` | network logits, shape `(B, L, V)` | `ReLF._backbone_forward` |
| `p_theta(w_l | xt, t)` | model posterior per position | `softmax(f_theta + s_E + logit_mask)` |
| `z` | ODE state at sampling time (paper: `Z_t`) | `generate_samples` |

`E` is a frozen (non-trainable buffer) embedding asset with a zero row for `[PAD]`; the headline is the d = 32 Hellinger-SVD asset `assets/flow_embed_hsvd32s.pt` (built by `scripts/train_embedding.py` + `build_schedule.py`; it also carries the unigram `pi` and the u -> alpha `schedule`).

## Definition of the ReLF

**Corruption (forward) process.** Linear interpolation in the fixed flow
space, independently per position:

    xt = t * x1 + (1 - t) * x0,   x0 ~ N(0, I)

so the conditional kernel is Gaussian and analytic:

    q_{t|1}(xt_l | w_l) = N(xt_l ; t * E[w_l], (1 - t)^2 * I_d)

The amplitude signal-to-noise ratio is `SNR(t) = t / (1 - t)`, i.e.
`log-SNR = logit(t)`; it diverges as `t -> 1`.

**Training objective.** Sample `t ~ p(t)` (the `training.t_sampling`
density: `uniform`, SD3-style `logit_normal`, or `arc_length` = the asset's
information arc-length schedule on [0, t_end], all via inverse-CDF on
stratified/antithetic uniforms), corrupt, and minimize per-position
cross-entropy against the true tokens:

    L = E_{t ~ p(t), w ~ data, x0} [ sum_l  -log p_theta(w_l | xt, t) ]

The model predicts `p_theta(w_l | xt, t)` **independently per position**
(the joint dependence enters only through the full corrupted sequence `xt`
feeding the transformer). Any weight over `t` enters **only as the sampling
density `p(t)`**, never as a CE multiplier.
The ReLF has **no likelihood interpretation** — no NLL/BPD/PPL of data
anywhere.

**Sampling (reverse) ODE.** The velocity field decomposes through the
posterior expectation of `x1` ("anticipated embedding"):

    x1_hat(z, t) = E_{p_theta}[x1 | z, t] = softmax(f_theta(z, t)) @ E
    v(z, t)      = (x1_hat(z, t) - z) / (1 - t)

integrated from `z ~ N(0, I)` at `t=0` (Euler on a uniform grid in the
simplest form; the headline sampler is the frozen 64-NFE `knots` schedule with
an SDE refresh); the final step returns
`argmax(logits)` instead of taking the last update.

## Singular Logits (paper technique — implemented behind `algo.singular_logits`)

This section documents the paper's "Singular Logits and Constraint
Enforcement" technique in this repo's notation. It is implemented in
`algo.py` behind the `algo.singular_logits` config flag (fp32 throughout,
on in every live config; the code default is `False`, which parameterizes
the logits **directly** as `f_theta(xt, t)`). The analytic term is formed
as `[a * xt, b] @ [E; ||E||^2]^T` (`ReLF._singular_inputs` /
`_singular_desc`, applied chunkwise by `_add_singular_logits`) and added
everywhere the logits define a distribution: the training CE (`singular`
argument of `chunked_linear_cross_entropy`, or `_exact_singular_ce` for
1 - t < 5e-3), the self-conditioning bootstrap
(`_expectation_from_features`), and every sampler. With
`algo.singular_unigram_reference` the reference `r_1` is the asset's unigram `pi`
instead of uniform: `log pi` is folded into `ReLF.logit_mask`.

### The problem: singular conditional marginals as t -> 1

The true posterior over the token at position `l` given the state is

    q_{1|t}(w_l | xt)  ∝  q_1(w_l | context) * q_{t|1}(xt_l | w_l)
                       ∝  q_1(.) * exp( -||xt_l - t * E[w_l]||^2 / (2 (1 - t)^2) )

As `t -> 1` the SNR `t/(1-t)` diverges, the Gaussian likelihood
concentrates, and the posterior collapses to a **one-hot** vector on the
nearest embedding. The true logits therefore diverge as `t -> 1`. A
transformer that parameterizes these logits directly must output unbounded
values, which its bounded-input/bounded-output nature resists — training
near `t = 1` becomes unstable.

### The fix: analytically separate the divergent part

Define the **singular logits** `s_E(xt, t)`, a closed-form function of the
state, time, and the fixed embedding matrix `E`, and let the network model
only the bounded residual:

    log p_theta(w_l | xt, t) := softmax-normalize over w of
        s_E_l(xt, t)[w] + f_theta_l(xt, t)[w]

i.e. total logits = analytic singular logits + network residual logits.

**Theorem (bounded residual).** Suppose there is a reference distribution
`r_1(w)` over sequences with `1/beta <= q_1(w) / r_1(w) <= beta` for some
`1 <= beta < inf`. Define the singular logits as the log-posterior under
the reference:

    s_E_l(xt, t)[w] := log r_{1|t}(w_l = w | xt)

Then the optimal residual is uniformly bounded:

    | f*_l(xt, t)[w] | = | log q_{1|t}(w_l = w | xt) - log r_{1|t}(w_l = w | xt) |
                       <= 2 log beta

so the network only ever has to represent values of magnitude `2 log beta`,
never the diverging component.

**Lemma (tractable form).** If the reference factorizes over positions,
`r_1(w) = prod_l r_1(w_l)`, the singular logits are per-position and
depend only on that position's state coordinate:

    s_E_l(xt, t)[w] = log[ r_1(w) * q_{t|1}(xt_l | w) ]
                      - log sum_{w~} [ r_1(w~) * q_{t|1}(xt_l | w~) ]

For **uniform** `r_1`, and plugging in the Gaussian kernel above, this is
(up to the per-position softmax normalizer, which cancels):

    s_E_l(xt, t)[w] = - ||xt_l - t * E[w]||^2 / (2 (1 - t)^2)

— a Gaussian-kernel score against every row of `E`, exactly computable
from the interpolation process. The `1/(1-t)^2` factor carries the entire
divergence as `t -> 1`; the residual `f_theta` stays O(1).

### Constraint enforcement: exact terminal rounding

**Proposition.** Let the trajectory follow `dz/dt = v(z, t)` with
`v = (x1_hat - z) / (1 - t)` and `x1_hat = softmax(logits) @ E`. If the
probabilities become one-hot as `t -> 1` (which the singular-logits
parameterization forces, since `s_E` dominates any bounded residual), then
each position `z_l` converges to a row of `E` exactly.

Consequence: a naive continuous model trained directly on discrete data
accumulates optimization error, and its terminal state lands only *near*
an embedding — rounding from embeddings to tokens introduces errors. With
singular logits, the flow is forced to "make a selection" from the
discrete token set at `t = 1`: terminal states coincide with valid token
embeddings and **rounding error is eliminated**. (The current repo
sidesteps terminal rounding differently: the last sampler step returns
`argmax(logits)` directly rather than reading off `z`.)

Two method-level remarks that follow from the math (not implementation
guidance):

- The analytic form of `s_E` above is specific to the linear interpolation
  `xt = t * x1 + (1 - t) * x0` with `x0 ~ N(0, I)`; a different schedule or
  noise law changes the kernel `q_{t|1}` and hence the closed form.
- Near `t = 0` the singular logits are nearly flat (`SNR ~ 0`), so the
  parameterization reduces to the direct one; the decomposition only
  changes what `f_theta` must represent at high `t`.

## Self-conditioning and distilled SC guidance (repo extensions)

**Self-conditioning** (`algo.self_cond`, Analog Bits). The network also sees its own previous
estimate: `f_theta(xt, t, x_sc)` with `x_sc = x1_hat` from the previous sampler step (zeros at step
0 = the "cold" mode). Training draws one no-grad cold pass at the same `(xt, t)`, giving
`x1_hat_u`, and feeds `k * x1_hat_u` with `k ~ Bernoulli(self_cond_p)` per example, so both the cold
and the warm mode are trained.

**Distilled self-conditioning guidance** (`algo.sc_guidance`, ELF). The network is also conditioned on a
per-example scale `w` (`1 + w ~ LogUniform[1 + omega_min, 1 + omega_max]`). A second no-grad pass in
the warm mode gives features `f_c`; with `f_u` from the cold pass, the CE is taken on logits shifted by
the stop-gradient term

    Delta = (1 - 1/w) * k * sg[f_c - f_u] W^T,     loss = CE(y, l_theta(xt, k x1_hat_u, t, w) - Delta)

at every t. The CE fixed point is

    l_warm(w) = l_cold + w * (log p_warm - l_cold)

i.e. logit-space classifier-free guidance between the cold and warm posteriors, distilled into one
forward. Inference is distilled only: `sampling.sc_guidance_scale = w` fed at every forward (1 NFE). The explicit
cold/warm extrapolation exists only inside training (the shift above). `w = 1` is plain self-conditioning.
