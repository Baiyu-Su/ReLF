"""Reflow (rectification) on top of standard flow training, switched on by `reflow.enabled`.

One yaml, one launch (`mode: train`), three stages:

1. collect -- the teacher (`reflow.teacher_path`, EMA weights) maps x0 ~ N(0, I) to tokens with the
   yaml's own deterministic sampler (the `sampling` block, `reflow.collect_steps` forwards at
   guidance scale `reflow.collect_sc_guidance_scale`). Pairs land as shards under
   `reflow.coupling_dir`; every rank takes a slice of the missing shards, existing shards are
   skipped (a preempted job re-enters and finishes only the rest), and a directory holding pairs
   of another teacher / sampler is refused (META.json check).
2. train -- the student starts from the teacher's EMA weights and trains with the standard
   objective, t density, optimizer, LR schedule and a fresh EMA. The only change is the coupling:
   each sequence's x0 is its stored source point (`ReLF.corrupt_continuous(x0=...)`) instead of a
   fresh N(0, I) draw.
3. interp (regularized reflow) -- theta = lam EMA(teacher) + (1 - lam) theta_student (`reflow.lam`,
   the paper's lambda: the weight pulled back toward the teacher; the student is the raw endpoint or
   the run's EMA, `reflow.interp_from`), written to
   `<save_dir>/interp/checkpoint-<step>` (model.safetensors + ema.pt holding theta,
   REFLOW_INTERP.json). The final eval reads these weights.

Shard k = `{'tokens': int32 (n, L), 'x0': fp16 (n, L, d), 'meta': {...}}`; x0 is drawn from a CPU
generator seeded `reflow.seed_base + k` (device-independent), standardized per flow dimension when
the sampler is the 'mh' state form with `eta_strat: std` (as the sampler would do to its own draw),
rounded to fp16, and the sampler runs from EXACTLY those fp16 values, so a stored pair reproduces
the teacher run.
"""
import json
import os
import time

import torch
from omegaconf import OmegaConf

from relf import utils

LOGGER = utils.get_logger(__name__)

META_KEYS = ('teacher_path', 'collect_steps', 'collect_sc_guidance_scale', 'seed_base', 'shard_size', 'sampler',
             'x0_standardized', 'flow_dim', 'length')


def enabled(config):
    rcfg = config.get('reflow', None)
    return bool(rcfg is not None and rcfg.get('enabled', False))


def num_shards(rcfg):
    n, s = int(rcfg.num_pairs), int(rcfg.shard_size)
    if n <= 0 or s <= 0 or n % s:
        raise ValueError(f'reflow.num_pairs ({n}) must be a positive multiple of reflow.shard_size ({s})')
    return n // s


def shard_path(rcfg, k):
    return os.path.join(rcfg.coupling_dir, f'shard_{k:05d}.pt')


def check_config(config):
    rcfg = config.reflow
    if config.algo.name != 'relf':
        raise ValueError('reflow.enabled needs algo.name relf')
    for key in ('teacher_path', 'coupling_dir'):
        if not rcfg.get(key, ''):
            raise ValueError(f'reflow.enabled needs reflow.{key}')
    if not os.path.isfile(os.path.join(rcfg.teacher_path, 'ema.pt')):
        raise FileNotFoundError(f'reflow.teacher_path has no ema.pt: {rcfg.teacher_path}')
    if not 0.0 <= float(rcfg.lam) < 1.0:
        raise ValueError(f'reflow.lam must be in [0, 1), got {rcfg.lam}')
    if str(rcfg.interp_from) not in ('raw', 'ema'):
        raise ValueError(f"reflow.interp_from must be 'raw' or 'ema', got {rcfg.interp_from}")
    num_shards(rcfg)


def init_from_teacher(model, config):
    """Copy the teacher checkpoint's EMA weights into `model`.

    They are both the student's start point and the weights the pairs are collected with.
    """
    from relf import trainer

    path = config.reflow.teacher_path
    ema = trainer.load_eval_checkpoint(model, path, device='cpu', disable_ema=False, decay=config.training.ema)
    if ema.num_updates == 0:
        # `_load_ema_or_reinit` silently re-initializes from the raw weights on a mismatch
        # (`num_updates` stays 0)
        raise RuntimeError(f'{path}/ema.pt did not load (EMA re-initialized from raw weights)')
    ema.copy_to(model.parameters())
    LOGGER.info(f'Reflow: student initialized from the EMA weights of {path}')


def _sampler_meta(config):
    return dict(grid=config.sampling.grid, **OmegaConf.to_container(config.eval_sampler, resolve=True))


def _shard_noise(seed, n, L, d, standardize):
    g = torch.Generator(device='cpu').manual_seed(seed)
    e = torch.randn((n, L, d), generator=g, dtype=torch.float32)
    if standardize:
        e = (e - e.mean(dim=1, keepdim=True)) / e.std(dim=1, keepdim=True).clamp_min(1e-6)
    return e


def _barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


@torch.no_grad()
def collect_pairs(model, config, device, rank, world_size):
    """Write the missing coupling shards (stage 1, see the module docstring).

    Every rank returns only once all `num_pairs // shard_size` shards exist. `model` holds the
    teacher's EMA weights.
    """
    rcfg = config.reflow
    n_shards = num_shards(rcfg)
    if model.sampling_solver == 'sde':
        raise ValueError('reflow collection needs a deterministic sampler (eval_sampler.solver sde is stochastic)')
    standardize = model.state_form == 'mh' and model.eta_strat == 'std'
    L, d = model.num_tokens, model.flow_dim
    meta = dict(
        teacher_path=os.path.abspath(rcfg.teacher_path), collect_steps=int(rcfg.collect_steps),
        collect_sc_guidance_scale=float(rcfg.collect_sc_guidance_scale), seed_base=int(rcfg.seed_base),
        shard_size=int(rcfg.shard_size), sampler=_sampler_meta(config), x0_standardized=bool(standardize),
        flow_dim=d, length=L,
    )
    os.makedirs(rcfg.coupling_dir, exist_ok=True)
    meta_path = os.path.join(rcfg.coupling_dir, 'META.json')
    # Every rank lists the missing shards BEFORE anyone writes, so the round-robin split agrees
    missing = [k for k in range(n_shards) if not os.path.isfile(shard_path(rcfg, k))]
    if rank == 0:
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                old = json.load(f)
            diff = {k: (old.get(k), meta[k]) for k in META_KEYS if old.get(k) != meta[k]}
            if diff:
                raise ValueError(f'{rcfg.coupling_dir} holds pairs of another collection (stored, current): {diff}')
        else:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
    _barrier()
    if not missing:
        LOGGER.info(f'Reflow: all {n_shards} shards present in {rcfg.coupling_dir}, skipping collection')
        return
    mine = missing[rank::world_size]
    LOGGER.info(f'Reflow: collecting {len(missing)} of {n_shards} shards ({len(mine)} on rank {rank}), '
                f'T={rcfg.collect_steps}, w={rcfg.collect_sc_guidance_scale}')
    batch, shard_size = int(rcfg.collect_batch_size), int(rcfg.shard_size)
    if shard_size % batch:
        raise ValueError(f'reflow.shard_size ({shard_size}) must be a multiple of reflow.collect_batch_size ({batch})')
    was_training, scale = model.training, model.sc_guidance_scale
    model.to(device)
    model.eval()
    model.sc_guidance_scale = float(rcfg.collect_sc_guidance_scale)
    try:
        for k in mine:
            t0 = time.time()
            x0 = _shard_noise(int(rcfg.seed_base) + k, shard_size, L, d, standardize).to(torch.float16)
            toks = torch.empty((shard_size, L), dtype=torch.int32)
            for s in range(0, shard_size, batch):
                z0 = x0[s:s + batch].to(device, torch.float32)
                ids = model.generate_samples(num_samples=z0.shape[0], num_steps=int(rcfg.collect_steps), z0=z0)
                toks[s:s + batch] = ids.to(torch.int32).cpu()
            out = shard_path(rcfg, k)
            tmp = f'{out}.tmp{rank}'
            torch.save({'tokens': toks, 'x0': x0, 'meta': dict(meta, shard=k, seed=int(rcfg.seed_base) + k)}, tmp)
            os.replace(tmp, out)
            print(f'[reflow collect] rank {rank}: {out} ({time.time() - t0:.0f} s)', flush=True)
    finally:
        model.sc_guidance_scale = scale
        model.train(was_training)
    # Wait on the filesystem, not in a collective: ranks finish up to one shard apart, which may
    # exceed the NCCL timeout for large models
    while any(not os.path.isfile(shard_path(rcfg, k)) for k in missing):
        time.sleep(15)
    _barrier()


class CouplingDataset(torch.utils.data.Dataset):
    """The first `num_pairs // shard_size` shards of `reflow.coupling_dir`.

    Items are `{'input_ids': int64 (L,), 'attention_mask': fp32 ones (L,), 'x0': fp16 (L, d)}` --
    the usual training columns plus the sequence's own source point, which `ReLF.forward` takes as
    `x0`. Shards are memory-mapped lazily per worker (`torch.load(mmap=True)`), so random access
    costs one page-in per item.
    """

    def __init__(self, rcfg, length, flow_dim):
        self.files = [shard_path(rcfg, k) for k in range(num_shards(rcfg))]
        missing = [f for f in self.files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f'{len(missing)} coupling shards missing, e.g. {missing[0]}')
        self.shard_size = int(rcfg.shard_size)
        first = torch.load(self.files[0], mmap=True, weights_only=False)
        if tuple(first['tokens'].shape) != (self.shard_size, length) or first['x0'].shape[-1] != flow_dim:
            raise ValueError(
                f'coupling shards are tokens {tuple(first["tokens"].shape)}, x0 d={first["x0"].shape[-1]}; '
                f'the run expects ({self.shard_size}, {length}), d={flow_dim}'
            )
        self._n = self.shard_size * len(self.files)
        self._cache = {}

    def __len__(self):
        return self._n

    def _shard(self, k):
        blob = self._cache.get(k)
        if blob is None:
            blob = torch.load(self.files[k], mmap=True, weights_only=False)
            self._cache[k] = blob
        return blob

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        k, j = divmod(int(idx), self.shard_size)
        blob = self._shard(k)
        ids = blob['tokens'][j].to(torch.int64)
        return {
            'input_ids': ids,
            'attention_mask': torch.ones(ids.shape[0], dtype=torch.float32),
            'x0': blob['x0'][j].clone(),
        }


@torch.no_grad()
def interpolate_and_save(trainer, config):
    """Interpolate teacher and student weights and save the result (stage 3).

    theta = lam EMA(teacher) + (1 - lam) theta_student goes into the model AND the run's EMA shadow
    (so every EMA-reading eval sees theta), then is saved under
    `<save_dir>/interp/checkpoint-<step>`.
    """
    rcfg = config.reflow
    lam, src = float(rcfg.lam), str(rcfg.interp_from)
    model = trainer.model
    params = [p for p in model.parameters() if p.requires_grad]
    teacher = torch.load(os.path.join(rcfg.teacher_path, 'ema.pt'), map_location='cpu', weights_only=True)
    teacher = teacher['shadow_params']
    if len(teacher) != len(params) or any(t.shape != p.shape for t, p in zip(teacher, params)):
        raise ValueError(f'{rcfg.teacher_path}/ema.pt does not match the model parameters')
    if src == 'ema':
        if trainer.ema is None:
            raise ValueError("reflow.interp_from 'ema' needs training.ema > 0")
        student = trainer.ema.shadow_params
    else:
        student = params
    theta = [
        (lam * t.to(device=p.device, dtype=torch.float64) + (1.0 - lam) * s.to(p.device, torch.float64)).to(p.dtype)
        for t, s, p in zip(teacher, student, params)
    ]
    for p, v in zip(params, theta):
        p.copy_(v)
    if trainer.ema is not None:
        trainer.ema.shadow_params = [v.clone() for v in theta]
    step = trainer.state.global_step
    out_dir = os.path.join(config.checkpointing.save_dir, 'interp', f'checkpoint-{step}')
    trainer.save_model(out_dir)
    if trainer.args.should_save:
        if trainer.ema is not None:
            torch.save(trainer.ema.state_dict(), os.path.join(out_dir, 'ema.pt'))
        with open(os.path.join(out_dir, 'REFLOW_INTERP.json'), 'w') as f:
            json.dump(dict(teacher=os.path.abspath(rcfg.teacher_path), student_step=step, student=src, lam=lam), f)
        print(f'[reflow interp] theta = {lam:.2f} EMA(teacher) + {1.0 - lam:.2f} {src}(student) -> {out_dir}',
              flush=True)
    _barrier()
