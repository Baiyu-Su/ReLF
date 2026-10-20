"""`ReLFTrainer`: `transformers.Trainer` with EMA, sampling-based eval, and ReLF checkpoint extras."""

import gc
import os

import safetensors.torch
import torch
import transformers
from omegaconf import ListConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from relf import metrics
from relf.model import ExponentialMovingAverage
from relf import optimizers


def eval_score_flags(config):
    """Return the (gen-ppl, entropy) switches of the sample scoring.

    `eval.compute_generative_perplexity` and `eval.compute_entropy` are independent. Both need
    samples, so `eval.generate_samples` gates the whole in-run / final eval (`main.py`'s
    `sample_eval` always samples). `compute_entropy` defaults to True for resolved configs written
    before the key existed.
    """
    return (
        bool(config.eval.compute_generative_perplexity),
        bool(config.eval.get('compute_entropy', True)),
    )


def _step_list(steps):
    if isinstance(steps, ListConfig):
        return list(steps)
    if isinstance(steps, int):
        return [steps]
    return list(steps)


def _to_eval_key(key, prefix):
    """Map a `namespace/name` (or bare `name`) metric key to `<prefix>_<name>`.

    transformers' `rewrite_logs` then routes it to the `<prefix>/` wandb namespace. Keys that don't
    start with `eval_`/`test_` would get a `train/` prefix instead (`train/val/...` panels).
    """
    name = key.rsplit('/', 1)[-1]
    return f'{prefix}_{name}'


class _TrainerCallback(transformers.TrainerCallback):
    """EMA maintenance and accumulation-counter bookkeeping."""

    def __init__(self, trainer):
        self._trainer = trainer

    def _reset_accumulation_counter(self):
        # ReLF only (t-window slicing); the AR baseline (`ar.GPT`) has no such counter
        model = self._trainer.model
        if hasattr(model, 'reset_accumulation_counter'):
            model.reset_accumulation_counter()

    def on_train_begin(self, args, state, control, **kwargs):
        if self._trainer.ema:
            self._trainer.ema.move_shadow_params_to_device(args.device)
        self._reset_accumulation_counter()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._reset_accumulation_counter()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._trainer.ema:
            self._trainer.ema.update(self._trainer.model.parameters())


class ReLFTrainer(transformers.Trainer):
    """`transformers.Trainer` with EMA and ReLF sampling-based eval.

    Evaluation swaps in EMA weights, generates samples with the configured sampler, scores
    generative perplexity and sample entropy, then restores the training weights. The flow model
    has no likelihood interpretation, so no NLL/PPL over the validation set is computed.
    """

    def __init__(self, relf_config, **kwargs):
        super().__init__(**kwargs)
        self.relf_config = relf_config
        if relf_config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=relf_config.training.ema
            )
        else:
            self.ema = None
        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=relf_config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=relf_config.eval.perplexity_batch_size,
        )
        self.metrics.to(self.args.device)
        self.add_callback(_TrainerCallback(self))

    def create_optimizer(self):
        """Build the `optim.name` optimizer (`adamw` or `muon`) via `optimizers.py`.

        The stock Trainer optimizer path is never used. Safe to override here: transformers defers
        optimizer creation to `train()`, long after `__init__` has set `relf_config`.
        """
        if self.optimizer is None:
            self.optimizer = optimizers.build_optimizer(self.model, self.relf_config)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Build the LR scheduler named by `trainer.lr_scheduler`.

        `warmup_stable_decay` builds the 1-sqrt-cooldown LambdaLR
        (`optimizers.build_wsd_scheduler`); every other value takes the stock path
        (`main._build_training_args`). Either way it is one LambdaLR over the (possibly Split-)
        optimizer's live `param_groups`, so the accelerate `prepare()` interaction is identical to
        the stock path.
        """
        if self.relf_config.trainer.get('lr_scheduler', 'constant_with_warmup') != 'warmup_stable_decay':
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            self.lr_scheduler = optimizers.build_wsd_scheduler(
                self.optimizer if optimizer is None else optimizer,
                self.relf_config,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _eval_mode(self):
        if self.ema and not self.relf_config.eval.disable_ema:
            print('Copying EMA parameters to model')
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
        else:
            print('No EMA parameters')
        self.model.eval()

    def _train_mode(self):
        if self.ema and not self.relf_config.eval.disable_ema:
            self.ema.restore(self.model.parameters())
        self.model.train()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        del eval_dataset, ignore_keys
        # Sampling allocates multi-GiB transients; release the training loop's cached allocator
        # segments first so a fragmented pool cannot strand otherwise-free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._eval_mode()
        self.metrics.reset()

        # The flow model has no likelihood interpretation, so eval does not touch the validation
        # set: it only generates samples and scores generative perplexity and sample entropy
        eval_metrics = {}
        if self.relf_config.eval.generate_samples:
            eval_metrics.update(self._sample_and_score())

        self._train_mode()
        self._release_eval_memory()

        # Route every metric to the `<prefix>/` (default `eval/`) wandb namespace. Our internal keys
        # use a `val/` prefix, which `rewrite_logs` would drop under `train/val/...`; normalize to
        # `<prefix>_<name>` instead.
        eval_metrics = {_to_eval_key(k, metric_key_prefix): v for k, v in eval_metrics.items()}

        self.log(eval_metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, eval_metrics
        )
        return eval_metrics

    def final_evaluate(self):
        """Run the post-training low-variance eval.

        Same protocol as `evaluate()` (EMA weights, sampling, gen-ppl + entropy) but over
        `sampling.num_final_eval_batches` sample batches per step count. Results go to the console
        log and the `final_eval/` wandb namespace, including a summary table.
        """
        config = self.relf_config
        num_batches = config.sampling.get('num_final_eval_batches', 8)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._eval_mode()
        self.metrics.reset()

        final_metrics = {}
        if config.eval.generate_samples:
            final_metrics = self._sample_and_score(
                num_sample_batches=num_batches, log_sample_table=False
            )

        self._train_mode()
        self._release_eval_memory()

        num_samples = num_batches * config.loader.eval_batch_size * self.args.world_size
        final_metrics = {f"final_eval/{k.rsplit('/', 1)[-1]}": v for k, v in final_metrics.items()}
        if self.args.process_index == 0:
            for num_steps in _step_list(config.sampling.steps):
                gen_ppl = final_metrics.get(f'final_eval/gen_ppl_T{num_steps}')
                entropy = final_metrics.get(f'final_eval/sample_entropy_T{num_steps}')
                print(
                    f'[final eval] T={num_steps} ({num_samples} samples): '
                    f'gen_ppl={gen_ppl}, sample_entropy={entropy}'
                )
            self._log_final_eval_table(final_metrics, num_samples)
        return final_metrics

    def _release_eval_memory(self):
        """Offload the gen-ppl scorer and return eval-time allocator segments to the driver.

        Sampling + the gen-ppl scorer (gpt2-large) carve the allocator pool into a layout the
        training step cannot reuse; without a release the first post-eval step re-allocates its full
        activation peak into a fragmented pool and can OOM (~10 GiB
        reserved-but-unallocated). `gc.collect()` runs before `empty_cache()` so a
        cyclic-referenced scorer model is actually dropped.
        """
        self.metrics.offload_eval_model()
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _log_final_eval_table(self, final_metrics, num_samples):
        import wandb

        if wandb.run is None:
            return
        rows = [
            [
                self.state.global_step,
                num_steps,
                num_samples,
                final_metrics.get(f'final_eval/gen_ppl_T{num_steps}'),
                final_metrics.get(f'final_eval/sample_entropy_T{num_steps}'),
            ]
            for num_steps in _step_list(self.relf_config.sampling.steps)
        ]
        table = wandb.Table(
            columns=['global_step', 'sampling_steps', 'num_samples', 'gen_ppl', 'sample_entropy'],
            data=rows,
        )
        wandb.log(
            {
                'final_eval/results': table,
                'train/global_step': self.state.global_step,
                **final_metrics,
            }
        )

    def _sample_and_score(self, num_sample_batches=None, log_sample_table=True):
        """Generate samples per configured step count and score them."""
        config = self.relf_config
        if num_sample_batches is None:
            num_sample_batches = config.sampling.num_sample_batches
        do_ppl, do_entropy = eval_score_flags(config)
        sample_metrics = {}
        for num_steps in _step_list(config.sampling.steps):
            self.metrics.gen_ppl.reset()
            self.metrics.sample_entropy.reset()
            current_text_samples = []

            for _ in range(num_sample_batches):
                samples = self.model.generate_samples(
                    num_samples=config.loader.eval_batch_size, num_steps=num_steps
                )
                if do_entropy:
                    self.metrics.record_entropy(samples)
                decoded_batch = self.model.tokenizer.batch_decode(samples)
                if len(current_text_samples) < config.sampling.num_sample_log:
                    current_text_samples.extend(decoded_batch)
                if do_ppl:
                    self.metrics.record_generative_perplexity(
                        decoded_batch, self.model.num_tokens, device=self.args.device
                    )

            if do_ppl:
                sample_metrics[f'val/gen_ppl_T{num_steps}'] = (
                    self.metrics.gen_ppl.compute().item()
                )
            if do_entropy:
                sample_metrics[f'val/sample_entropy_T{num_steps}'] = (
                    self.metrics.sample_entropy.compute().item()
                )

            if log_sample_table and self.args.process_index == 0:
                self._log_sample_table(num_steps, current_text_samples)
        return sample_metrics

    def _log_sample_table(self, num_steps, text_samples):
        import wandb

        if wandb.run is None:
            return
        log_samples = text_samples[: self.relf_config.sampling.num_sample_log]
        table = wandb.Table(columns=['Generated Samples'], data=[[s] for s in log_samples])
        wandb.log({f'eval/samples_T{num_steps}@global_step{self.state.global_step}': table})

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            output_dir = os.path.join(
                self._get_output_dir(trial=trial),
                f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}',
            )
            if self.ema:
                torch.save(self.ema.state_dict(), os.path.join(output_dir, 'ema.pt'))
            # The t-stratification generator is a shared stream (identical on all ranks), so rank
            # zero's copy is authoritative. ReLF only: the AR baseline has no t stream.
            if hasattr(self.model, 't_rng_state'):
                torch.save(
                    {'t_generator': self.model.t_rng_state()},
                    os.path.join(output_dir, 't_rng.pt'),
                )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self.ema:
            ema_path = os.path.join(resume_from_checkpoint, 'ema.pt')
            self.ema = _load_ema_or_reinit(
                ema_path, self.model, self.relf_config.training.ema, self.args.device
            )
        rng_path = os.path.join(resume_from_checkpoint, 't_rng.pt')
        if not hasattr(self.model, 'set_t_rng_state'):
            return  # AR baseline: no t stream to restore
        if os.path.isfile(rng_path):
            state = torch.load(rng_path, map_location='cpu', weights_only=True)
            self.model.set_t_rng_state(state['t_generator'])
        else:
            print(
                f'[WARNING] No t-generator state at {rng_path} - '
                'the t stratification stream restarts from the seed'
            )


def _load_ema_or_reinit(ema_path, model, decay, device):
    """Load EMA state; on any mismatch re-initialize from model weights."""
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    if not os.path.isfile(ema_path):
        print(f'[WARNING] No EMA state at {ema_path} - initializing EMA from model weights')
        return ema
    try:
        state = torch.load(ema_path, map_location=device, weights_only=True)
        current_params = [p for p in model.parameters() if p.requires_grad]
        shadow = state['shadow_params']
        if len(shadow) == len(current_params) and all(
            s.shape == p.shape for s, p in zip(shadow, current_params)
        ):
            ema.load_state_dict(state)
            # The yaml's decay wins over the saved one, so a branch resumed from another run's
            # checkpoint can change `training.ema` (a same-yaml resume is unaffected)
            if ema.decay != decay:
                print(f'[EMA] decay {ema.decay} from {ema_path} -> {decay} (training.ema)')
                ema.decay = decay
        else:
            print('[WARNING] EMA shape mismatch - reinitializing from model weights')
    except Exception as e:
        print(f'[WARNING] Failed to load EMA state: {e} - reinitializing from model weights')
    return ema


def load_eval_checkpoint(model, checkpoint_path, device, disable_ema=False, decay=0.9999):
    """Load an HF Trainer checkpoint dir into `model` for eval.

    Returns the loaded EMA (or None). Does not copy EMA weights into the model; callers decide when
    to swap.
    """
    safe_path = os.path.join(checkpoint_path, 'model.safetensors')
    bin_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
    if os.path.isfile(safe_path):
        state_dict = safetensors.torch.load_file(safe_path, device='cpu')
    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f'No model weights found under {checkpoint_path}')
    model.load_checkpoint_state_dict(state_dict)
    model.to(device)
    if disable_ema:
        return None
    return _load_ema_or_reinit(os.path.join(checkpoint_path, 'ema.pt'), model, decay, device)
