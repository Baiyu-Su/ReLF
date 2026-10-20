---
name: relf-embedding-assets
description: Building, inspecting, and using ReLF flow-embedding assets (assets/flow_embed_*.pt) — the train_embedding.py -> build_schedule.py chain, the asset file contract (embedding, pi, schedule, frozen gauge), recovering the build configs stored inside an asset, and the data pitfalls of the cached OpenWebText windows. Use when creating or modifying an embedding asset or its time schedule, changing flow_dim, wiring an asset into a training config, or writing any script that counts tokens from the OWT cache.
---

# ReLF flow-embedding assets

The ReLF flow space is a frozen token embedding `E` (`x1 = E[w]`). An asset fixes *where* in SNR each
token distinction resolves; its time schedule decides how the training-t density and the inference grid
spend their budget there. The two are built together and ship in one file.

## 1. Asset contract

`assets/flow_embed_<name>s.pt` (the trailing `s` = carries a schedule) is a `torch.save` dict:

| key | content |
|---|---|
| `embedding` | `(50258, d)` fp32: 50257 GPT-2 tokens + an all-zero `[PAD]` row appended last |
| `pi` | `(50257,)` unigram distribution of the statistics slice |
| `schedule` | `alpha_of_u` (fp64 uniform-u -> alpha lookup), `alpha_end`, `t_end`, the curves `grid_alpha / r / s / u`, and the schedule build `config` |
| `config` | the embedding build config (yaml string) |
| `gauge`, `mode`, `source`, `tau_bigram`, `tau_probe`, `selected`, `perm_control_median_cost` | build metadata and audit |

- **Gauge: pi-weighted zero mean and pi-weighted average power P = d. Never rescale or re-center an
  asset**, in a script or in the model: the schedule's `t_end` and arc-length lookup are computed at
  exactly this scale and silently stop matching otherwise. `t = alpha / (1 + alpha)`; storing alpha keeps
  `1 - t` exact in fp64.
- The model reads `embedding`, `pi`, `schedule.alpha_of_u` and `schedule.t_end` (`relf/algo.py`). The
  head is padded to 50304 for GEMMs; padded slots get zero embedding rows and are masked to -inf.
- A training config wires an asset with `model.token_embed_path` and a matching `model.flow_dim = d`
  (only `input_proj`'s input width depends on d). `training.t_sampling: arc_length`, `sampling.grid:
  arc_length` / `knots`, and (with uniform t sampling) `training.t_max: 't_end'` all read the asset's
  schedule.

## 2. Build chain (single GPU, corpus statistics only, no neural network)

```bash
python scripts/train_embedding.py --config-name <embed_cfg>       # -> outputs/<run>/<out_name> (bare asset)
cp outputs/<run>/<out_name> assets/                               # e.g. flow_embed_hsvd32.pt
python scripts/build_schedule.py --config-name <schedule_cfg>     # -> NEW assets/<out_asset> (…s.pt)
python scripts/train_embedding.py --self-test                     # CPU checks of the estimators
```

1. **`train_embedding.py`** — adjacent-pair count tables on disjoint SHAPE / EVAL slices of the cached
   OWT windows (chunks `i % 4 == 3` are EVAL), bigram smoothing `tau` chosen by held-out next-token log
   loss, then the R=1 both-neighbor **Hellinger-SVD** init (the dense V x C matrix is never formed),
   duplicate-row jitter, and the pi gauge. `mode`:
   - `hellinger_svd` — export the init as is (the shipped assets);
   - `svd_collision` — Adam steps on the collision barrier only;
   - `force_field` — commit-cost force field on the top-N confusable universe.

   Every asset records a held-out nearest-neighbor substitution-cost audit, the endpoint identifiability
   gate (`alpha_end`) and a permutation control.
2. **`build_schedule.py`** — rebuilds the embedding run's count tables deterministically (same seed and
   draw order, from the embed config named by `embed_config`) and checks them against the asset's `pi`,
   then computes the conditional-channel information curves r_c (bigram-conditional) and r_u (unigram),
   the endpoint `alpha_end` (entropy and top-1 error tolerances under the unigram prior), and the
   normalized arc length u of (r_c, r_u). It writes a **new** file `out_asset`; the input asset is never
   modified (a run may be training on it) and an asset that already has a schedule is refused.

**Rule: a new asset gets its schedule before any training run uses it.** Testing a new embedding on
uniform-time defaults confounds the geometry with a schedule mismatch; a uniform-time run is fine only as a
deliberate second ablation, never as the primary readout.

## 3. Recovering build configs from an asset

No `embed_*` / `schedule_*` configs ship; each asset stores both recipes. To rebuild or vary the shipped
asset, write them back out (the schedule config names the embed config and the bare asset it expects):

```python
import torch
blob = torch.load('assets/flow_embed_hsvd32s.pt', map_location='cpu', weights_only=False)
open('configs/embed_hsvd32.yaml', 'w').write(blob['config'])               # embedding recipe
open('configs/schedule_hsvd32.yaml', 'w').write(blob['schedule']['config'])  # schedule recipe
```

Then set `data.cache_dir` to your OWT cache. For a different dimension, change `flow_dim`, `power`
(P = d), `run_name`, `out_name`, and the schedule config's `embed_config` / `asset_path` / `out_asset`
together. `build_schedule.py` takes the seed and data slices from the embed config it names, so leave that
config unchanged once its asset is built, or the schedule's `pi` check fails.

## 4. OpenWebText cache pitfalls

- Both scripts read the tokenized cache `openwebtext-{train,valid}_*_bs1024_wrapped.dat` under
  `data.cache_dir` and refuse to build it; run any training config once (the dataloader builds and caches
  it) before building assets.
- **The cached train windows contain rare stray token ids >= 50257** (observed: about one per 10^9 tokens). Never `assert tok.max() < V` — whether it fires depends on which
  windows a strided sample happens to hit. Filter instead and print the count: drop pairs touching an
  out-of-vocab id when counting (`count_adjacent`, `count_both_neighbors`), drop whole windows for
  validation tokens.
- `blob['pi']` is stored fp32 while the rebuild is fp64, so the schedule's `pi` check uses a 1e-7
  tolerance. An asset whose `pi` came from a different copy of the OWT cache can raise it with `pi_tol` in
  the schedule config.

## 5. Changing assets in a training config

Switching `model.token_embed_path` or `flow_dim` is a new experiment (new yaml and `run_name`; see the
`relf-experiments` skill), never a resume. Checkpoints load strict, so a checkpoint is tied to the
`flow_dim` it was trained with. `algo.singular_unigram_reference` folds the asset's `log pi` into the
logits, so it also ties a run to its asset.
