---
name: relf-experiments
description: How to configure, launch, resume, validate, and evaluate ReLF / AR-baseline experiments in this repo — the one-yaml-per-experiment convention, per-run isolation, the validation ladder before a real launch, and the sample_eval -> make_eval_configs -> summarize_evals workflow (gen-ppl + entropy). Use when adding or changing a config, launching or resuming a training run, deciding how much to test a change, or running/tabulating evaluations.
---

# Running ReLF experiments

Method math lives in the `relf` skill; asset building in `relf-embedding-assets`. This skill is the
workflow around them.

## 1. Configs: one standalone yaml per experiment

- Every experiment is a single self-contained file `configs/<run_name>.yaml`, selected with
  `--config-name <run_name>`. No Hydra config groups (`model=small` is invalid): every setting is inlined.
- **The yaml is the only source of experiment settings.** A real run launches bare:

  ```bash
  torchrun --standalone --nproc_per_node=4 -m main --config-name relf_s    # train
  python -m main --config-name <eval_cell_yaml>                            # sample_eval
  ```

  No `key=value` CLI overrides and no environment-variable switches for anything that changes behavior
  (e.g. `torch.compile` is `model.compile` in the yaml, not an env flag). The resolved config is dumped
  into the run folder, so the run folder alone must fully describe the run. Overrides are acceptable only
  for throwaway debugging (section 3).
- Values the user controls (batch sizes, accumulation, node / GPU counts, step counts) are explicit
  literals, not interpolations. `get_datasets` asserts
  `global_batch_size == batch_size * num_nodes * num_gpus * accumulate_grad_batches`.
- A new experiment = copy its parent yaml, change `run_name` to the file name, change the deltas, and
  state the deltas in the header comment (`# relf_x: relf_s with ... (relf_s: ...)`). A sweep is several
  files, not one file plus overrides.
- `ReLF.__init__` reads its config keys directly (no code-side defaults), so a new knob must be added to
  every live yaml. Inference-only sampler variants go in the `eval_sampler` section (defaults = plain warm
  Euler).
- Parameterization switches (`algo.singular_logits`, `algo.singular_unigram_reference`, `algo.self_cond`,
  `algo.sc_guidance`) change what the network outputs: whole-run only, never toggled mid-run or at eval.
- Step counts are optimizer steps. `seed` is a fixed literal; it controls RNG, never run identity.

## 2. Every run is isolated

- Output folder: `outputs/<run_name>_<date>_<id>/` with `id` = `SLURM_JOB_ID`, else
  `TORCHELASTIC_RUN_ID`, else a timestamp (`hydra.run.dir`). Checkpoints, samples, and the resolved config
  live inside it; Slurm logs go to `logs/<run_name>_<jobid>.out/err`.
- The wandb run id is never set (no `id` key, no `id=` to `wandb.init`): every launch, including a resume,
  is a new wandb run. Two submissions must never share an output folder, log file, or wandb run.
- When touching a launcher, config, or logging path, check every per-run surface (folder, logs,
  checkpoints, wandb, sample dumps) is keyed by the run id before calling it ready.
- **Resume is explicit only**: `checkpointing.resume_from_ckpt: true` plus `resume_ckpt_path` pointing at
  a specific `checkpoints/checkpoint-<step>` (a null path raises; there is no auto-detection of the latest
  checkpoint). A resumed run still gets a fresh folder and wandb run. Never resume unless asked.
- Checkpoints: HF dirs `checkpoint-<step>/` (`model.safetensors`, optimizer / scheduler / RNG state,
  `trainer_state.json`) plus `ema.pt` and `t_rng.pt` (the training-t generator).
- Read run statistics from the local copies, never the wandb remote: final values in
  `outputs/<run>/wandb/run-*/files/wandb-summary.json`, history in `files/output.log` or
  `checkpoints/checkpoint-*/trainer_state.json`.
- Experiment results (tables, logs, figures) never go into the repo; `results/` is gitignored.

## 3. How much to test a change

| Change | Required check |
|---|---|
| yaml value of an existing, validated knob (lr, depth, batch, steps); docs; comments; renames | syntax only (`python -m py_compile`, `git diff --check`); do not launch a GPU run — say it was not run |
| new or changed training-path code: loss, optimizer, trainer / callbacks, model forward, data path | a short REAL training run before any multi-GPU / cluster launch |
| sampler-only code | a short `sample_eval` on an existing checkpoint, compared against the previous output |

The real training run goes through the full stack:

```bash
WANDB_MODE=disabled python -m main --config-name <cfg> \
    trainer.max_steps=300 trainer.warmup_steps=10 loader.global_batch_size=<small> loader.batch_size=<small>
```

and must show the train loss decreasing. Interface-level unit tests are not enough: `transformers.Trainer`
/ accelerate `prepare()` round-trip optimizer state and rebuild param-group dicts, so a component that is
correct in isolation can silently do nothing in the real loop. The signature of that failure is a flat loss
with a healthy grad norm and a normally ramping logged lr. Put debug runs under a throwaway
`outputs/_dbg_<name>/`.

Known non-bugs: with the zero-init adaLN and head, the hidden matrices get zero gradient for the first
steps from a fresh init (model dynamics, not an optimizer bug).

## 4. Evaluation

- Metrics are **generative perplexity** (gpt2-large scorer on the model's own samples) and **unigram
  entropy** of the samples, read together (lower gen-ppl at collapsed entropy is not an improvement). The
  ReLF has no likelihood: never report NLL / BPD / perplexity of data. The same holds for the AR baseline
  (`algo.name: ar`) — its role is a gen-ppl vs entropy curve on its own samples (temperature sweep);
  never compute a held-out NLL for it, even as a diagnostic.
- Training runs log an in-run eval (EMA weights) every `trainer.val_check_interval` steps and, with
  `eval.do_final_eval`, a lower-variance final eval after training.
- Checkpoint evaluation is one bare `sample_eval` yaml per (checkpoint, cell), generated from the
  training config so the model section always matches the checkpoint:

  ```bash
  python scripts/make_eval_configs.py --train-config relf_s --run-glob 'outputs/relf_s_2*' \
      --steps 80000 --cells euler_T64,euler_T256,sde
  python -m main --config-name eval_relf_s_ck80000_euler_T64      # one process per cell
  python scripts/summarize_evals.py --md eval_table.md            # tabulate outputs/eval_*/samples.json
  ```

  Default cell: EMA weights, 16 x 256 = 4096 samples, seed 1. `--shards K` splits a cell into K processes
  (`summarize_evals.py --pool` merges them); `--seed` gives fresh-seed confirmations. The AR baseline uses
  `scripts/make_ar_eval_configs.py` (one yaml per checkpoint x temperature).
- Sampler knobs: `sampling.grid` (`uniform`, `arc_length`, `knots`), `eval_sampler.solver` (`euler`,
  `sde`, `heun` = 2 NFE/step on the uniform grid), `sampling.sc_guidance_scale` (distilled guidance, 1 NFE).
  Compare samplers at equal NFE.
- Step budget: practical sampling uses at most 1024 steps (256 / 1024 are the reference regimes). Finer
  grids are only reference solutions for measuring discretization error; do not recommend settings that
  only pay off beyond 1024 steps.

## 5. Hardware notes

Per-GH200 (96 GB) batch size: 128 for training, 256 for inference-only `sample_eval` (~38 GiB peak with
the row-chunked sampler). TF32 stays off process-wide (`main.py`); scripts that bypass `main.py` must not
enable it. Everything outside the network forward (integrator state, time grid, softmax expectation) is
fp32.
