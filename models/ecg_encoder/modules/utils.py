import contextlib
from typing import Optional, Tuple
import numpy as np
import torch


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def batch_by_size(
    indices,
    num_tokens_fn,
    num_tokens_vec = None,
    max_tokens = None,
    max_signals = None,
    required_batch_size_multiple = 1
):
    """Yield mini-batches of indices. Batches may contain sequences of
    different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        required_batch_size (int, optional): require batch size to be less than
            N
    """
    try:
        from src.dataset.data_utils_fast import (
            batch_by_size_fn,
            batch_by_size_vec
        )
    except ImportError:
        raise ImportError(
            "Please build Cython components with: "
            "`python setup.py build_ext --inplace`"
        )
    except ValueError:
        raise ValueError(
            "Please build (or rebuild) Cython components with `python setup.py build_ext --inplace`."
        )

    max_tokens = (
        int(max_tokens) if max_tokens is not None else -1
    )
    max_signals = max_signals if max_signals is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype = np.int64, count = -1)
    
    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype = np.int64, count = -1)

    if num_tokens_vec is None:
        return batch_by_size_fn(
            indices,
            num_tokens_fn,
            max_tokens,
            max_signals,
            bsz_mult
        )
    else:
        return batch_by_size_vec(
            indices,
            num_tokens_vec,
            max_tokens,
            max_signals,
            bsz_mult
        )

def get_buckets(sizes, num_buckets):
    buckets = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation = 'lower'
        )[1:]
    )
    return buckets

def get_bucketed_sizes(orig_sizes, buckets):
    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes
