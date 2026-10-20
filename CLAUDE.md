# CLAUDE.md

Rectified Language Flow (ReLF).
Style: `black -l 100 --skip-string-normalization`; ~100-120 char lines.
Method math (notation, objective, sampling ODE, Singular Logits) lives in the **`relf` skill**
(`.claude/skills/relf/SKILL.md`); experiment workflow in `relf-experiments`, embedding assets in
`relf-embedding-assets`. None of it is repeated here.
Headline config: `relf_s` (scaled up as `relf_m` / `relf_l`; reflow from it in `relf_s_reflow`).

## Method (repo-specific only)

- **Flow space.** `x1 = E[w]`, frozen `(V, flow_dim)` buffer `ReLF.token_embed` from
  `model.token_embed_path` (headline: `assets/flow_embed_hsvd32s.pt`, d = 32); each asset carries `pi`
  and the `schedule`. Learnable `DIT.input_proj`: flow_dim -> hidden. Asset scale is frozen — never
  rescale it.
- **t sampling.** `training.t_sampling`: `uniform` (`t_min`/`t_max`; `t_max: 't_end'` from the asset
  schedule), `logit_normal`, or `arc_length` (the asset's arc-length schedule, t in [0, t_end]). The t
  weight is only this density, never a CE multiplier — train loss is NOT comparable across densities.
- **Inference grid.** `sampling.grid`: `uniform`, `arc_length` (equal-info-arc-length Euler grid from the
  asset's schedule, decode at `t_end`), or `knots` (fixed-endpoint piecewise-linear knot map,
  `eval_sampler.grid_a/grid_b/grid_knots`). `eval_sampler.solver`: `euler` (default), `sde` (any grid),
  or `heun` (2 NFE/step, uniform grid only).
- **Singular logits** (`algo.singular_logits`, code default False, on in every shipped config): adds the
  analytic Gaussian-posterior term to network logits, so the network models only a bounded residual.
  `algo.singular_unigram_reference` folds the asset's log pi into `logit_mask` (requires
  singular_logits; parameterization change — whole-run only).
- **Self-conditioning** (`algo.self_cond`, `self_cond_p`): the backbone also consumes its previous
  x1_hat through a widened zero-init `input_proj`. Training: one no-grad cold bootstrap pass at the same
  (x_t, t), kept per example with prob `self_cond_p`; sampling: step 0 cold, then the previous step's raw
  x1_hat (`eval_sampler.carry_mode` warm / cold).
- **Distilled self-conditioning guidance** (`algo.sc_guidance`, `omega_min/max`): the network is
  conditioned on a per-example scale w (`sc_w_embed`); a second no-grad warm pass forms the stop-grad
  feature shift (1 - 1/w)·keep·(f_warm - f_cold) the CE subtracts (`hidden_shift`), so one forward at
  scale w approximates cold/warm logit-space guidance. Sampling: `sampling.sc_guidance_scale` = w at every
  forward (1 NFE). Self-cond and sc_guidance are parameterization changes — whole-run only.
- **Precision.** Everything outside the network forward is fp32 (bf16 sampling collapses at high step
  counts); final step is `argmax`. TF32 is off process-wide (`main.py`; scripts that skip `main.py` must
  not turn it on).
- **Padded head** (`model.vocab_padded_size`, default 50304): head rounded up from real vocab 50258
  (50257 BPE + `[PAD]`) for GEMM; padded slots have zero `token_embed` and are masked -inf via
  `ReLF.logit_mask` (zero gradient). Checkpoints load strict (unpadded ones are not supported).

## Scope

ReLF training/sampling, small DiT, OpenWebText scripts, the AR GPT baseline (`ar.py`; gen-ppl +
entropy only, never NLL), and reflow behind `reflow.enabled` (`reflow.py`).
The interpolant stays `t*x1 + (1-t)*x0` in plain `t`; a LUT may only shape the t-sampling density or
grid placement, never warp time inside the interpolant.

## Environment

Install with `pip install -r requirements.txt` (see `README.md`). Keep debug/validation output under the
gitignored `outputs/` (e.g. `outputs/_dbg_<run>/`).

## Entry Points

Train multi-GPU `torchrun --standalone --nproc_per_node=N -m main` (via `transformers.Trainer`);
eval single-process `python -m main`. `main.py` is the only root module; the library lives in the
`relf/` package (`from relf import algo`); scripts put the repo root on `sys.path`.

- `main.py` — Hydra entry; modes train and `sample_eval` (`load_eval_checkpoint` on a
  `checkpoint-XXXX`).
- `algo.py` — `ReLF(SamplerMixin, nn.Module)`: asset, t density, corruption, loss; config stored as
  `self.relf_config` (not `.config`, which the Trainer reads as a `PretrainedConfig`).
  `forward(input_ids, attention_mask, x0=None)` returns `{'loss'}` with no `**kwargs`, so the Trainer
  normalizes by grad-accum.
- `fused_ce.py` — `chunked_linear_cross_entropy` (fused head + CE + singular logits, also used by `ar.py`);
  `samplers.py` — `SamplerMixin` (`generate_samples`, euler / sde / heun; `state_form` z / mh) + grid /
  gate helpers. Both row-chunk logits in fp32 — no full `(B*L, V)` / `(B, L, V)` tensor is ever
  materialized.
- `ar.py` — the autoregressive GPT baseline (`algo.name: ar`), dispatched by `main.py`; same block
  dimensions as the DiT, gen-ppl + entropy eval only.
- `trainer.py` — `ReLFTrainer`: EMA on `on_optimizer_step`; `evaluate()` empties the CUDA cache,
  swaps in EMA, samples, scores gen-ppl + entropy (`metrics.Metrics`, rank-synced) under `eval/`,
  restores. `create_optimizer` delegates to `optimizers.py`.
- `optimizers.py` — `build_optimizer` on `optim.name`: `adamw` or `muon`. Muon takes every 2D
  weight except `input_proj` and the LM head, which stay on AdamW with the norms/biases;
  membership is by explicit name regex with an exhaustiveness check — NEVER `ndim==2`. Both groups
  share `optim.lr` and one schedule; `SplitOptimizer.param_groups` MUST stay a live property —
  accelerate's `prepare()` rebuilds the sub-optimizer group dicts (see its docstring).
- `model.py` — DiT backbone, time via `TimestepEmbedder`; JiT/DiT init with zero-init adaLN +
  head, so from fresh init the hidden matrices get zero grad for the first steps (model dynamics,
  not an optimizer bug). Also holds `ExponentialMovingAverage` (the EMA `trainer.py` drives).
- `reflow.py` — `reflow.enabled` runs (`configs/relf_s_reflow.yaml`): student init from the teacher
  EMA, sharded pair collection in the train launch (existing shards skipped, META.json mismatch refused),
  `CouplingDataset` (the `x0` column `ReLF.forward` takes), final interp -> `<run>/interp/checkpoint-<step>`.
- `dataloader.py` — GPT-2 tokenizer; OWT cached under `data.cache_dir`; asserts batch-size divisibility.
  Cached train windows hold rare stray tokens >= 50257: filter and count them, never `assert tok.max() < V`.

## Seeding

Two RNG streams from `config.seed`, both checkpointed and restored on resume:

- `ReLF._t_rng`: bare seed on **every** rank, never offset — the global t stratification needs an
  identical stream. One stratified vector of `global_batch_size` per grad-accum window; each rank
  slices its chunk.
- Global CUDA RNG: offset `seed + process_index` in `main._train` (noise, dropout, eval sampling
  decorrelate); CPU seed shared (sampler sync).

## Configuration

One standalone yaml per experiment in `configs/` (`relf_s` / `relf_m` / `relf_l`, `relf_s_reflow`);
each file's header states its deltas vs its parent. Picked bare via `--config-name`.
Settings inlined — no Hydra groups (`model=small` invalid). `ReLF.__init__` reads its keys directly (no
code defaults); inference-only sampler variants live in the `eval_sampler` section (defaults = plain warm
Euler). Step counts are optimizer steps;
`trainer.lr_scheduler` `constant_with_warmup` (default), `cosine_with_min_lr`, or `warmup_stable_decay`.
`hydra.run.dir` resolves to `SLURM_JOB_ID` -> `TORCHELASTIC_RUN_ID` -> timestamp. `model.compile` =
single global torch.compile switch. Per-GPU batch size (96 GB): 128 for training, 256 for
inference-only `sample_eval` (~38 GiB peak with the row-chunked sampler).

## Experiment Hygiene (mandatory)

- **YAML is the single source of experiment settings**: real runs launch bare — selection via
  `--config-name` only, never nested overrides (`trainer.max_steps=100` is throwaway debug) or env
  knobs. The resolved config is dumped into the run folder.
- **Every Slurm submission is isolated, keyed by `SLURM_JOB_ID`**: `outputs/<run>_<date>_<jobid>/`,
  `logs/<run>_<jobid>.out/err`, auto-generated wandb id (never pass `id`; even a resume gets a new
  run).
- **Resume only on explicit request**: `resume_from_ckpt: true` + `resume_ckpt_path` at a
  `checkpoint-XXXX` (null path raises); a resumed run still gets a fresh job-id folder + wandb run.
- `seed` is a fixed literal; it controls RNG, never run identity.
- **Read stats from local copies, never the wandb remote**: final values in
  `outputs/<run>/wandb/run-*/files/wandb-summary.json`; history in `files/output.log` or
  `checkpoints/checkpoint-*/trainer_state.json`.

## Scripts

Asset build chain: `train_embedding.py` (`mode: hellinger_svd` -> hsvd32; also `force_field`,
`svd_collision`) -> `build_schedule.py` (writes a NEW `*s.pt` asset with `blob['schedule']`; the input
asset is never modified). The build configs are stored inside each asset (`relf-embedding-assets` skill).
Eval tooling: `make_eval_configs.py`, `make_ar_eval_configs.py`, `summarize_evals.py`. Kernel test:
`test_sc_guidance_kernel.py`.

## Checkpoints

HF dirs `checkpoints/checkpoint-<step>/` (`model.safetensors`, optimizer/scheduler/rng state,
`trainer_state.json`) plus `ema.pt` and `t_rng.pt` (t-generator). Lightning `.ckpt` not loadable;
compile checkpoints may carry `._orig_mod.` keys, stripped by `filter_compile_state_dict`.

## Validation

Syntax checks (`python -m py_compile`, `bash -n`, `git diff --check`) suffice for trivial edits
(yaml values on validated paths, docs, renames). A non-trivial change to optimizer, loss, or
trainer/callbacks gets a short REAL train run before any sbatch: `python -m main --config-name <cfg>`
with throwaway overrides (`trainer.max_steps=300 trainer.warmup_steps=10`, small single-GPU batch,
`WANDB_MODE=disabled`), confirming train loss decreases. Interface-level smoke tests do not cross
Trainer/accelerate `prepare()`, which mutates optimizer internals.
