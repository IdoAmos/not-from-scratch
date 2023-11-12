###### Description:
# This file contains the functions for different sequence masking strategies.

import torch
import numpy as np

def span_mask(x: torch.Tensor, masking_ratio, span_length):
    """
    Masking spans of length span_length in the sequence without overlapping. Fraction of masked
    tokens will match masking_ratio.
    :param x: input sequence [..., sequence_length, 1]
    :param masking_ratio: masking ratio
    :param span_length: length of the span to be masked
    :return: masked sequence [..., sequence_length, 2] - last channel is binary mask (1-masked, 0-unmasked)
    """

    assert masking_ratio < 1.0, 'masking ratio must be less than 1.0'

    # retrieve sequence length and finder number of spans to mask
    seq_length = x.shape[-2]    # sequence length
    if span_length < 1:
        span_length = int(seq_length * span_length)
    assert span_length < seq_length, 'span length must be less than sequence length'
    num_pos = int((seq_length * masking_ratio) / span_length)  # number of spans

    # vars to be used in sampling spans
    mask = np.zeros_like(x)  # will hold the mask
    available_indices = np.arange(seq_length - span_length)  # sample indices from here
    probs = np.ones_like(available_indices, dtype=np.float32)  # will hold the probs for sampling - allows ignoring indices

    # sample spans
    for i in range(num_pos):
        # get random indices to mask
        start = np.random.choice(available_indices, p=probs / probs.sum())
        end = start + span_length

        # mask the selected indices
        mask[..., start:end, 0] = 1

        # zero out probabilities of overlapping spans
        start = max([0, start - span_length])
        probs[start: end] = 0

    # apply mask
    mask = torch.from_numpy(mask)

    return mask