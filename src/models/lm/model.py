import string
import torch
import torch.nn as nn
from src.models.sequence.base import SequenceModule

from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers import AutoTokenizer

from torchtext.vocab.vocab import Vocab as TorchTxTVocab
from functools import partial
from .utils import batch_mappings

# should match the model names in src.utils.registry
LM_MODEL_LIST = ['pythia_lm']

def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    assert head_mask is None and (attention_mask is None or attention_mask.all()), "Mask is not supported"

    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    is_causal = query_length == key_length
    attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    return attn_output, None

def _causal_attn(self, query, key, value, attention_mask=None, head_mask=None):
    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    assert head_mask is None and (attention_mask is None or attention_mask.all()), "Mask is not supported"

    attn_output = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    return attn_output, None

@torch.no_grad()
def replace_attn(m):
    if type(m) == GPTNeoXAttention:
        m._attn = partial(_attn, m)

@torch.no_grad()
def replace_causal_attn(m):
    if type(m) == GPTNeoXAttention:
        m._attn = partial(_causal_attn, m)

PYTHIA_D_MODEL = 512

class PythiaWrapper(SequenceModule):
    """
    A wrapper around huggingface langauge models to adhere to repos trainer interface.
    Seperate the embedding, model and head.
    Args:
        decoder: the outputheads mapping to cls, pass to model to enforce dtype matching, Pythia uses float16
        flash_attn: replace the attention with a faster version
        src_tokenizer: the tokenizer used for the input, used to map the input to the model tokens
        txt_input: whether the input is text or not, if not, the input is expected to be integers
    """
    def __init__(self, flash_attn=True, src_tokenizer=None, batch2tensor_fn=None, causal=False, pretrained=True,
                 use_pythia_embeddings=True, **kwargs):
        super(PythiaWrapper, self).__init__()

        self.model_id = 'EleutherAI/pythia-70m-deduped'
        self.cache_dir = './cache'
        self.pretrained = pretrained
        self.use_pythia_embeddings = use_pythia_embeddings

        if self.pretrained:
            self.model = AutoModel.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self.config = self.model.config

            self.src_tokenizer = src_tokenizer
            self.txt_input = True if src_tokenizer is not None else False
        else:
            self.config = AutoConfig.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self.model = AutoModel.from_config(self.config)

        self.d_model = self.config.hidden_size
        self.d_output = self.d_model
        self.causal = causal
        self.dtype = self.config.torch_dtype

        if flash_attn:
            if self.causal: self.model.apply(replace_causal_attn)
            else: self.model.apply(replace_attn)

        if self.pretrained:  # for matching the input tokens to pythia tokens
            if self.txt_input:
                self._map_batch2str = self._map_batch_tokens2str
                self._insert_special_tokens()
            else:
                # expects inputs to be integers, e.g. images
                self._map_batch2str = self._map_batch_int2str

            self.batch2tensor = self._map_batch2tensor
            if batch2tensor_fn is not None:
                # batch2tensor maps from string batch back to tensors, requires a tokenizer
                self.batch2tensor = batch_mappings[batch2tensor_fn]

    def _reverse_src_tokenizer(self, x):
        return self.src_tokenizer.lookup_tokens(x)

    def _map_batch_int2str(self, batch_input):
        return [list(map(str, x)) for x in batch_input.tolist()]

    def _map_batch_tokens2str(self, batch_input):
        return list(map(self._reverse_src_tokenizer, batch_input.tolist()))

    def _map_batch2tensor(self, batch, **kwargs):
        """
        Map a batch of strings to a tensor.
        """
        batch_output = [self.tokenizer(sample, return_tensors='pt').input_ids.squeeze(-1) for sample in batch]
        batch_output = torch.stack(batch_output, dim=0)
        return batch_output

    def apply_batch_mapping(self, batch_input):
        """
        Apply the mapping from a batch input.
        """
        assert batch_input.dtype == torch.long, f"Input should be long for mapping but got:{batch_input.dtype}"
        assert batch_input.ndim == 2, f"Input should be batched sequences but got:{batch_input.ndim}"

        # nested list with of strings with shape as batch
        batch_as_str = self._map_batch2str(batch_input)
        batch_output = self.batch2tensor(batch_as_str, tokenizer=self.tokenizer)

        # in case of mapping from old to new tokens - modify the special ones
        if self.txt_input:
            for mapping in self._special_token_mapping:
                batch_output[batch_output == mapping[0]] = mapping[1]

        return batch_output

    def _insert_special_tokens(self):
        """
        Find special tokens in src tokenizer that might not exist in Pythia tokenizer
        """
        # for mapping <pad>, <bos>, <eos> back to original values
        self._special_token_mapping = []
        for i in range(len(self.src_tokenizer)):
            s = self.src_tokenizer.lookup_tokens([i])[0]
            if s.startswith('<') and s.endswith('>'):
                new_tokens = {s[1:-1] + '_token': s}  # expected format for Pythia is e.g 'mask_token': <mask>
                self.tokenizer.add_special_tokens(new_tokens)

                # map the new tokens to the ones pythia trained on, e.g. <eos> -> <|endoftext|>
                new_token_val = self.tokenizer([s]).input_ids[0][0]
                if s[1: -1] in ['eos', 'bos', 'unk']:
                    src_val = self.tokenizer('<|endoftext|>').input_ids[0]
                    mapping = (new_token_val, src_val)
                    self._special_token_mapping.append(mapping)
                elif s[1: -1] in 'pad':
                    src_val = self.tokenizer('<|padding|>').input_ids[0]
                    mapping = (new_token_val, src_val)
                    self._special_token_mapping.append(mapping)
                else:
                    pass

    def forward(self, x, state=None, **kwargs):
        """
        Wraps the forward method of LLM with a mapping to match the encoders
        """
        # map input to tokens matching the model
        if self.use_pythia_embeddings:
            if self.pretrained:
                input_device = x.device
                x = self.apply_batch_mapping(x)
                x = x.to(input_device)

            h = self.model(x).last_hidden_state
        else:
            h = self.model(inputs_embeds=x)

        return h, None