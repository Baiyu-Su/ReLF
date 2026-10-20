"""OpenWebText data: GPT-2 tokenizer, packed fixed-length windows, and batch-size checks.

Adapted from the Flow Map Language Models codebase, Apache 2.0.
"""

import functools
import itertools
import os

import datasets
import tokenizers
import torch
import transformers

from relf import reflow
from relf import utils

LOGGER = utils.get_logger(__name__)


def _group_texts(examples, block_size, bos, eos):
    """Concatenate docs into [BOS] + (block_size - 2) tokens + [EOS] windows; drop the tail."""
    concatenated_examples = list(itertools.chain(*examples['input_ids']))
    total_length = len(concatenated_examples)
    new_block_size = block_size - 2
    total_length = (total_length // new_block_size) * new_block_size

    input_ids = []
    attention_masks = []
    for i in range(0, total_length, new_block_size):
        input_ids.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        attention_masks.append(torch.ones(block_size))
    return {'input_ids': input_ids, 'attention_mask': attention_masks}


def get_dataset(
    dataset_name,
    tokenizer,
    mode,
    cache_dir,
    insert_eos=True,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
):
    """Return OpenWebText packed into `block_size` windows ([BOS] ... [EOS]), cached on disk."""
    if dataset_name not in {'openwebtext-train', 'openwebtext-valid'}:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    eos_tag = '' if insert_eos else '_eosFalse'
    filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
    path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(path):
        LOGGER.info(f'Loading data from: {path}')
        return datasets.load_from_disk(path).with_format('torch')

    LOGGER.info(f'Generating new data at: {path}')
    split = 'train[:-100000]' if dataset_name == 'openwebtext-train' else 'train[-100000:]'
    data = datasets.load_dataset(
        'openwebtext',
        split=split,
        cache_dir=cache_dir,
        streaming=False,
        num_proc=num_proc,
        trust_remote_code=True,
    )

    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    bos_id = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'
        tokens = tokenizer(
            example['text'],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        if insert_eos:
            tokens = {'input_ids': [ids + [eos_id] for ids in tokens['input_ids']]}
        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc='Tokenizing',
    )
    tokenized_dataset = tokenized_dataset.remove_columns('text')

    group_texts = functools.partial(_group_texts, block_size=block_size, bos=bos_id, eos=eos_id)
    chunked_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True, desc='Grouping'
    )
    chunked_dataset.save_to_disk(path)
    return chunked_dataset.with_format('torch')


def get_tokenizer(config):
    """Return the GPT-2 tokenizer with BOS/EOS wrapping (`BertProcessing`) and an added `[PAD]`."""
    if config.data.tokenizer_name_or_path != 'gpt2':
        raise ValueError(f'Unsupported tokenizer: {config.data.tokenizer_name_or_path}')

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path)
    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(
        tokenizer, transformers.GPT2Tokenizer
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def get_datasets(config, tokenizer):
    """Return `(train_set, valid_set)` after checking the batch sizes against the visible GPUs."""
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        per_step = (
            config.loader.batch_size
            * config.trainer.num_nodes
            * num_gpus
            * config.trainer.accumulate_grad_batches
        )
        if config.loader.global_batch_size != per_step:
            raise ValueError(
                f'loader.global_batch_size {config.loader.global_batch_size} != batch_size '
                f'{config.loader.batch_size} x num_nodes {config.trainer.num_nodes} x {num_gpus} gpus x '
                f'accumulate_grad_batches {config.trainer.accumulate_grad_batches} = {per_step}'
            )
        if config.loader.eval_global_batch_size % num_gpus != 0:
            raise ValueError(
                f'loader.eval_global_batch_size {config.loader.eval_global_batch_size} '
                f'not divisible by {num_gpus} gpus.'
            )

    if reflow.enabled(config):
        # Reflow: the stored (x0, tokens) couplings collected from `reflow.teacher_path` (reflow.py)
        train_set = reflow.CouplingDataset(config.reflow, config.model.length, config.model.flow_dim)
        LOGGER.info(f'Reflow coupling data: {len(train_set)} pairs from {config.reflow.coupling_dir}')
    else:
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode='train',
            insert_eos=config.data.insert_train_eos,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            num_proc=config.loader.num_workers,
        )

    valid_set = get_dataset(
        config.data.valid,
        tokenizer,
        mode='validation',
        cache_dir=config.data.cache_dir,
        insert_eos=config.data.insert_valid_eos,
        block_size=config.model.length,
        num_proc=config.loader.num_workers,
    )

    return train_set, valid_set
