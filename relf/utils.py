"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import collections
import functools
import logging
import os

import fsspec


def _is_rank_zero():
    return int(os.environ.get('RANK', 0)) == 0


def rank_zero_only(fn):
    """Run `fn` only on the global rank zero process."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if _is_rank_zero():
            return fn(*args, **kwargs)
        return None

    return wrapped


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initialize a multi-GPU-friendly Python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Wrap every level in `rank_zero_only`; otherwise each GPU process would repeat every log line
    for level in ('debug', 'info', 'warning', 'error', 'exception', 'fatal', 'critical'):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def config_choice(cfg, key, options):
    """Return `cfg[key]`, which must be one of `options` (the dispatch would send a typo to its else
    branch)."""
    value = cfg[key]
    if value not in options:
        raise ValueError(f'{key} must be one of {options}, got {value!r}')
    return value


def filter_compile_state_dict(state_dict):
    """Strip the torch.compile `._orig_mod.` infix from checkpoint state_dict keys."""
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('._orig_mod.', '.')
        new_state_dict[new_key] = v
    return new_state_dict
