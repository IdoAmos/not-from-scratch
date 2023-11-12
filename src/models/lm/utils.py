"""
Specific utilities for language models.
"""

import torch
from torch import nn

listops_mapping = {'[MAX': 'A', '[SM': 'B', '[MED': 'C', '[MIN': 'D'}

def map_sample(sample, mapping):
    return [s if s not in mapping.keys() else mapping[s] for s in sample]

def listops_batch_mapping_an(batch, tokenizer, **kwargs):
    """
    Map ListOps batch to match the model input.
    Mapping replaces special ListOps tokens with generic ones, e.g. '[MAX' -> 'A'

    Batch should be a list of strings, i.e. already converted from tokens back to the string sequence with the
    native ListOps tokenizer. Can be done with dataset.vocab, dataset = ListOps(_name_='ListOps')
    Returns:
        batch: tensor of shape (batch_size, seq_len)
    """
    batch = [tokenizer(map_sample(sample, listops_mapping), return_tensors='pt').input_ids.squeeze(-1) for sample in batch]
    batch = torch.stack(batch, dim=0)
    return batch

def listops_batch_mapping(batch, tokenizer, **kwargs):
    """
    Map ListOps batch to match the model input.
    Mapping replaces special ListOps tokens with the tokenization, e.g. '[MAX' -> [ '[' ,'MAX' ]
    Then pads accordingly to create a batch

    Batch should be a list of strings, i.e. already converted from tokens back to the string sequence with the
    native ListOps tokenizer. Can be done with dataset.vocab, dataset = ListOps(_name_='ListOps')
    And applies additional padding to create a batch of equal length.
    Returns:
        batch: tensor of shape (batch_size, seq_len)
    """
    batch = [tokenizer(sample).input_ids for sample in batch]

    tensor_batch = []
    for i, sample in enumerate(batch):
        tensor_batch.append(torch.tensor([item for sublist in sample for item in sublist]))

    padding_value = tokenizer(tokenizer.pad_token).input_ids[0]
    tensor_batch = torch.nn.utils.rnn.pad_sequence(tensor_batch, batch_first=True, padding_value=padding_value)
    return tensor_batch
    
batch_mappings = {"listops_an": listops_batch_mapping_an,  # replace operators with annonymous tokens
                  "listops": listops_batch_mapping}  # keep operators as tokens