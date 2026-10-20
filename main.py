"""Hydra entry point: training and `sample_eval` for the flow model and the AR baseline."""

import json
import os
from datetime import datetime

import fsspec
import hydra
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

from relf import algo, ar, dataloader, metrics, reflow, trainer, utils
from relf.trainer import ReLFTrainer

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)


@utils.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Print the Hydra config as a Rich tree.

    `resolve` resolves interpolations in the printed tree; `save_cfg` also writes the tree and the
    resolved yaml into `checkpointing.save_dir`.
    """
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        save_dir = config.checkpointing.save_dir
        with fsspec.open(f'{save_dir}/config_tree.txt', 'w') as fp:
            rich.print(tree, file=fp)
        # Persist the exact (resolved) config into the run folder, so the run is reproducible from
        # its own artifacts
        with fsspec.open(f'{save_dir}/config.yaml', 'w') as fp:
            fp.write(omegaconf.OmegaConf.to_yaml(config, resolve=True))


@utils.rank_zero_only
def _print_model_arch(model):
    """Log the parameter name/shape list and total parameter count."""
    print('Model architecture (name: shape):')
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        suffix = '' if p.requires_grad else '  [frozen]'
        print(f'  {name}: {tuple(p.shape)}{suffix}')
    print(f'Total parameters: {total:,}')


@utils.rank_zero_only
def _print_batch(train_set, valid_set, tokenizer, k=64):
    for ds_type, ds in [('train', train_set), ('valid', valid_set)]:
        print(f'Printing {ds_type} dataset sample.')
        sample = ds[0]['input_ids']
        print('input_ids.shape', tuple(sample.shape))
        first = sample[:k]
        last = sample[-k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def _maybe_init_wandb(config):
    """Pre-initialize the wandb run (rank zero) so the Trainer reuses it.

    Returns whether the Trainer should report to wandb (True on every rank when configured).
    """
    if config.get('wandb', None) is None:
        return False
    if int(os.environ.get('RANK', 0)) != 0:
        return True
    import wandb

    # Never pass an explicit run id: wandb auto-generates a unique one per init, so every
    # submission (including resumes) is a separate wandb run
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.get('name'),
        group=config.wandb.get('group'),
        job_type=config.wandb.get('job_type'),
        notes=config.wandb.get('notes'),
        tags=list(config.wandb.get('tags', [])),
        config=omegaconf.OmegaConf.to_object(config),
    )
    return True


def _build_training_args(config, report_to_wandb):
    if config.trainer.precision not in {'bf16', '32', 32}:
        raise ValueError(f'Unsupported precision: {config.trainer.precision}')
    lr_scheduler = config.trainer.get('lr_scheduler', 'constant_with_warmup')
    lr_scheduler_kwargs = {}
    if lr_scheduler == 'cosine_with_min_lr':
        lr_scheduler_kwargs = {'min_lr_rate': float(config.trainer.lr_min_ratio)}
    elif lr_scheduler == 'warmup_stable_decay':
        # Built by `ReLFTrainer.create_scheduler` (1-sqrt cooldown); the stock type below is unused
        lr_scheduler = 'constant_with_warmup'
    elif lr_scheduler != 'constant_with_warmup':
        raise ValueError(f'Unsupported trainer.lr_scheduler: {lr_scheduler}')
    return transformers.TrainingArguments(
        output_dir=os.path.join(config.checkpointing.save_dir, 'checkpoints'),
        max_steps=config.trainer.max_steps,
        per_device_train_batch_size=config.loader.batch_size,
        per_device_eval_batch_size=config.loader.eval_batch_size,
        gradient_accumulation_steps=config.trainer.accumulate_grad_batches,
        learning_rate=config.optim.lr,
        adam_beta1=config.optim.beta1,
        adam_beta2=config.optim.beta2,
        adam_epsilon=config.optim.eps,
        weight_decay=config.optim.weight_decay,
        # `trainer.lr_scheduler`: 'constant_with_warmup' (default, the flow runs) or
        # 'cosine_with_min_lr' (warmup -> cosine decay to `trainer.lr_min_ratio` x lr at max_steps)
        # or 'warmup_stable_decay' (warmup -> constant -> 1-sqrt cooldown over
        # `trainer.decay_steps` to `optim.final_lr`; `optimizers.build_wsd_scheduler`)
        lr_scheduler_type=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_steps=config.trainer.warmup_steps,
        max_grad_norm=config.trainer.gradient_clip_val,
        bf16=config.trainer.precision == 'bf16',
        eval_strategy='steps',
        eval_steps=config.trainer.val_check_interval,
        logging_steps=config.trainer.log_every_n_steps,
        save_strategy='steps',
        save_steps=config.checkpointing.save_every_n_steps,
        save_total_limit=None,
        seed=config.seed,
        dataloader_num_workers=config.loader.num_workers,
        dataloader_pin_memory=config.loader.pin_memory,
        dataloader_persistent_workers=config.loader.num_workers > 0,
        ddp_find_unused_parameters=False,
        report_to=['wandb'] if report_to_wandb else [],
        run_name=config.wandb.get('name') if report_to_wandb else None,
    )


def _build_model(config, tokenizer):
    """Build the model `algo.name` selects: 'relf' (the flow model) or 'ar' (the GPT baseline)."""
    if config.algo.name == 'relf':
        return algo.ReLF(config, tokenizer=tokenizer)
    if config.algo.name == 'ar':
        return ar.GPT(config, tokenizer=tokenizer)
    raise ValueError(f'Invalid algorithm name: {config.algo.name}')


def _load_eval_model(config, tokenizer):
    model = _build_model(config, tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ema = trainer.load_eval_checkpoint(
        model,
        config.eval.checkpoint_path,
        device=device,
        disable_ema=config.eval.disable_ema,
        decay=config.training.ema,
    )
    return model, ema


def _generate_samples(config, logger, tokenizer):
    """Sample from `eval.checkpoint_path` at each step count, score, and dump the samples json."""
    logger.info('Starting Sample Eval.')
    model, ema = _load_eval_model(config, tokenizer)
    if ema is not None:
        logger.info('Copying EMA parameters into the model.')
        ema.copy_to(model.parameters())
    else:
        logger.info('Sampling without EMA.')
    model.eval()

    eval_metrics = metrics.Metrics(
        gen_ppl_eval_model_name_or_path=config.eval.gen_ppl_eval_model_name_or_path,
        eval_ppl_batch_size=config.eval.perplexity_batch_size,
    )
    eval_metrics.to(model.device)
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Metrics are computed PER step count (T-major loop, accumulators reset per T): pooling several
    # T into one accumulator would make a T-sweep unreadable
    step_list = trainer._step_list(config.sampling.steps)
    do_ppl, do_entropy = trainer.eval_score_flags(config)
    per_step_metrics = {}
    for num_steps in step_list:
        eval_metrics.gen_ppl.reset()
        eval_metrics.sample_entropy.reset()
        for _ in range(config.sampling.num_sample_batches):
            samples = model.generate_samples(
                num_samples=config.loader.eval_batch_size, num_steps=num_steps
            )
            if do_entropy:
                eval_metrics.record_entropy(samples)
            text_samples = tokenizer.batch_decode(samples)
            if do_ppl:
                eval_metrics.record_generative_perplexity(
                    text_samples, config.model.length, device=model.device
                )
            all_samples.extend(list(text_samples))
        # A switched-off metric is stored as None (the samples.json keys stay fixed)
        per_step_metrics[num_steps] = {
            'generative_ppl': eval_metrics.gen_ppl.compute().item() if do_ppl else None,
            'entropy': eval_metrics.sample_entropy.compute().item() if do_entropy else None,
        }
        print(
            f'T={num_steps}: gen_ppl={per_step_metrics[num_steps]["generative_ppl"]}, '
            f'entropy={per_step_metrics[num_steps]["entropy"]}'
        )

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = per_step_metrics[step_list[-1]]['generative_ppl']
    entropy = per_step_metrics[step_list[-1]]['entropy']
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump(
            {
                # Top-level pair = the LAST step count (so a single-T config reports its only T);
                # the sweep lives in `per_step_metrics`. `generated_seqs` are T-major blocks of
                # `num_sample_batches` x `eval_batch_size` samples, in `step_list` order.
                'generative_ppl': generative_ppl,
                'entropy': entropy,
                'per_step_metrics': {str(k): v for k, v in per_step_metrics.items()},
                'generated_seqs': all_samples,
            },
            f,
            indent=4,
        )
    print('Samples saved at:', samples_path)


def _train(config, logger, tokenizer):
    logger.info('Starting Training.')
    report_to_wandb = _maybe_init_wandb(config)

    use_reflow = reflow.enabled(config)
    if use_reflow:
        reflow.check_config(config)
    else:
        train_set, valid_set = dataloader.get_datasets(config, tokenizer)
        _print_batch(train_set, valid_set, tokenizer)

    if config.training.get('finetune_path', ''):
        # A run either resumes its own checkpoint (`resume_from_ckpt`) or starts a reflow student
        # from its teacher (`reflow.teacher_path`). Refuse yamls that still set `finetune_path`
        # instead of silently training from scratch.
        raise ValueError('training.finetune_path is no longer supported; use resume_ckpt_path or reflow')
    model = _build_model(config, tokenizer)
    _print_model_arch(model)

    args = _build_training_args(config, report_to_wandb)
    if use_reflow:
        # reflow.py: teacher EMA weights -> student init; collect the missing (x0, tokens) pairs
        # with them on every rank; then the coupling dataset replaces the OWT train split
        reflow.init_from_teacher(model, config)
        reflow.collect_pairs(model, config, args.device, args.process_index, args.world_size)
        train_set, valid_set = dataloader.get_datasets(config, tokenizer)
        _print_batch(train_set, valid_set, tokenizer)
    trainer = ReLFTrainer(
        relf_config=config,
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=transformers.default_data_collator,
        processing_class=tokenizer,
    )

    # Per-rank RNG decorrelation: corruption noise, dropout, and eval sampling draw from the global
    # CUDA RNG, which `set_seed` (re-run identically inside `Trainer.__init__`) left in lockstep on
    # every rank. Offset it by rank so those draws differ across ranks. The CPU seed stays shared
    # (the shuffling sampler must agree across ranks), and `_sample_t` is unaffected: it has its own
    # shared-seed generator because the global t stratification needs an identical stream on all
    # ranks. On resume, the Trainer restores the saved per-rank CUDA RNG states over this.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + trainer.args.process_index)

    resume_ckpt = None
    if config.checkpointing.resume_from_ckpt:
        # Resume must be explicit: no auto-detection of the latest checkpoint
        resume_ckpt = config.checkpointing.resume_ckpt_path
        if resume_ckpt is None:
            raise ValueError(
                'checkpointing.resume_from_ckpt=true requires '
                'checkpointing.resume_ckpt_path to point at a checkpoint-XXXX dir.'
            )
        if not utils.fsspec_exists(resume_ckpt):
            raise FileNotFoundError(f'resume_ckpt_path does not exist: {resume_ckpt}')
        logger.info(f'Resuming training from: {resume_ckpt}')

    trainer.train(resume_from_checkpoint=resume_ckpt)

    if use_reflow:
        # The only departure from standard training: blend the student into the teacher's EMA. The
        # final eval below then reads the blended weights.
        reflow.interpolate_and_save(trainer, config)

    if config.eval.get('do_final_eval', False):
        logger.info('Running final evaluation with EMA weights.')
        trainer.final_evaluate()


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Run training or `sample_eval`, per `config.mode`."""
    transformers.set_seed(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name not in {'relf', 'ar'}:
        raise ValueError(f'Invalid algorithm name: {config.algo.name}')
    kwargs = {'config': config, 'tokenizer': tokenizer, 'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # TF32 OFF, process-wide. The network runs in bf16 autocast, which ignores this flag; the only
    # fp32 GEMMs are the tiny `DIT.input_proj` (flow_dim -> hidden) and the time-embedder MLP, plus
    # the sampler / loss singular and softmax-expectation matmuls, which rely on this flag (algo.py
    # has no local toggles). With TF32 on in training only, those two GEMMs ran at 10-bit mantissa
    # in training but in true fp32 in `generate_samples`: same weights, different features
    # (measured: ~1% argmax flips per position at t=0.5, ~8% of sampled tokens along a 256-step
    # trajectory; gen-ppl/entropy unchanged within noise). A constant flag also avoids a Dynamo
    # recompile on every eval entry and exit. Throughput cost: nil.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
