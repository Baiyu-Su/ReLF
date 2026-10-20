"""Generative perplexity and sample entropy, scored by an external GPT-2 family LM.

Adapted from the Flow Map Language Models codebase, Apache 2.0.
"""

import os
import typing

import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from transformers import logging

logging.set_verbosity_error()


class Perplexity(torchmetrics.aggregation.MeanMetric):
    """Token-weighted mean of per-token NLLs, exponentiated.

    Used only for *generative* perplexity: the NLL of generated samples scored
    by an external eval LM, not a likelihood of the flow model itself.
    """

    def update(
        self,
        value: typing.Union[float, torch.Tensor],
        weight: typing.Union[float, torch.Tensor] = 1.0,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)
        if value.numel() == 0:
            return
        self.mean_value += value.sum()
        self.weight += weight.sum()

    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    """Gen-ppl and sample-entropy accumulators plus the lazily loaded, cached scorer LM."""

    def __init__(self, gen_ppl_eval_model_name_or_path=None, eval_ppl_batch_size=None) -> None:
        # The scorer is always a GPT-2 family LM (gpt2-large): 1024-token context, EOS-masked NLL
        if 'gpt2' not in str(gen_ppl_eval_model_name_or_path).lower():
            raise ValueError(f'gen-ppl scorer must be a GPT-2 model, got {gen_ppl_eval_model_name_or_path!r}')
        self.gen_ppl = Perplexity()
        self.sample_entropy = torchmetrics.aggregation.MeanMetric()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(gen_ppl_eval_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._eval_model = None
        self._eval_model_device = None

    def _get_eval_model(self, device):
        """Return the scorer LM on `device`, loaded once on first use and kept (frozen, eval mode).

        Reloading it per call costs ~2.6 s per sample batch: negligible for one in-run eval,
        dominant when scoring thousands of batches.
        """
        if self._eval_model is None:
            self._eval_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.gen_ppl_eval_model_name_or_path
            ).eval()
        if self._eval_model_device != torch.device(device):
            self._eval_model = self._eval_model.to(device)
            self._eval_model_device = torch.device(device)
        return self._eval_model

    def offload_eval_model(self):
        """Move the cached scorer to CPU (kept loaded; re-uploaded by the next `_get_eval_model`).

        Training runs call this after each in-run eval so the 3 GiB fp32 scorer does not sit in GPU
        memory next to the training step.
        """
        if self._eval_model is not None and self._eval_model_device != torch.device('cpu'):
            self._eval_model = self._eval_model.to('cpu')
            self._eval_model_device = torch.device('cpu')

    def to(self, *args, **kwargs):
        self.gen_ppl = self.gen_ppl.to(*args, **kwargs)
        self.sample_entropy = self.sample_entropy.to(*args, **kwargs)

    def reset(self):
        self.gen_ppl.reset()
        self.sample_entropy.reset()

    @torch.no_grad()
    def _eval_retokenize(self, text_samples, max_length, device):
        """Retokenize generated text samples for the (GPT-2 family) eval model.

        Returns the input ids, their attention mask, and the eval context size (1024), all for the
        eval model.
        """
        samples = self.tokenizer(
            text_samples,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        attn_mask = samples['attention_mask'].to(device)
        samples = samples['input_ids'].to(device)
        return samples, attn_mask, 1024

    @torch.no_grad()
    def record_entropy(self, tokens):
        """Record each sample's unigram token entropy (nats) into `sample_entropy`."""
        for sample in tokens:
            _, counts = torch.unique(sample, return_counts=True, sorted=False)
            entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
            self.sample_entropy.update(entropy)

    @torch.no_grad()
    def record_generative_perplexity(
        self,
        text_samples: typing.List[str],
        max_length: int,
        retokenize: bool = True,
        device='cuda',
    ) -> None:
        """Score samples with the eval LM and accumulate their token-weighted NLL into `gen_ppl`.

        Every non-EOS token counts, plus the first EOS of each sequence; later EOS tokens (the
        padding, since pad = EOS) are masked out.
        """
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        eval_model = self._get_eval_model(device)
        if retokenize:
            samples, attn_mask, eval_context_size = self._eval_retokenize(
                text_samples, max_length=max_length, device=device
            )
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(device)
            eval_context_size = samples.shape[-1]
        batch_size = min(self.eval_ppl_batch_size, samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size : (i + 1) * batch_size], eval_context_size, dim=-1
            )
            _attn_mask = torch.split(
                attn_mask[i * batch_size : (i + 1) * batch_size], eval_context_size, dim=-1
            )
            for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                logits = eval_model(
                    sample_chunk.to(device), attention_mask=attn_mask_chunk.to(device)
                )
                logits = logits[0].transpose(-1, -2)
                nlls = F.cross_entropy(
                    logits[..., :-1], sample_chunk[..., 1:], reduction='none'
                )
                first_eos = (sample_chunk == self.tokenizer.eos_token_id).cumsum(-1) == 1
                token_mask = sample_chunk != self.tokenizer.eos_token_id
                valid_tokens = first_eos[..., 1:] + token_mask[..., 1:]
                self.gen_ppl.update(nlls * valid_tokens, valid_tokens)
