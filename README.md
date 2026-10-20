# ReLF

Rectified Language Flow (ReLF) on OpenWebText: tokens are embedded into a frozen
low-dimensional flow space, a DiT learns the posterior over tokens along the linear interpolant
`x_t = t x_1 + (1 - t) x_0`, and sampling integrates the induced ODE/SDE, decoding by argmax at the end.
On top of the base ReLF: singular logits with a unigram reference, self-conditioning, distilled
self-conditioning guidance in logit space, and regularized reflow. An autoregressive (AR) baseline (`ar.py`)
shares the stack for comparison.
The data pipeline and metrics are adapted from the Flow Map Language Models codebase ([arXiv:2602.16813](https://arxiv.org/abs/2602.16813), Apache 2.0); the DiT backbone follows [JiT](https://github.com/LTH14/JiT) ([arXiv:2511.13720](https://arxiv.org/abs/2511.13720), MIT).

## Install

```bash
pip install -r requirements.txt
# plus, built for your CUDA stack (see the notes in requirements.txt): flash-attn-4 (from source,
# --no-build-isolation) and quack-kernels (0.5.3 in the reference env)
```

## Run

Every experiment is one standalone yaml in `configs/`, selected bare (no CLI overrides for real runs):

```bash
torchrun --standalone --nproc_per_node=4 -m main --config-name relf_s   # train
python -m main --config-name <eval_cell_yaml>                                 # sample_eval (gen-ppl + entropy)
```

Eval-cell yamls come from `scripts/make_eval_configs.py`.

## Where things are

- `main.py` (Hydra entry, train / `sample_eval`); the `relf/` package: `algo.py` (`ReLF`: loss, t
  sampling), `samplers.py` (inference samplers), `fused_ce.py` (chunked head + CE), `model.py` (backbone),
  `trainer.py` (EMA, in-run eval), `optimizers.py` (Muon/AdamW split, WSD), `reflow.py`, `ar.py`,
  `dataloader.py`, `metrics.py`, `utils.py`.
- Method math: `.claude/skills/relf/SKILL.md`.
