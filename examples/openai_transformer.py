# original file: https://github.com/allenai/allennlp/blob/master/allennlp/modules/openai_transformer.py
# only selected modules copied and annotated in this file
# some parts re-written to clarify tensor shapes

# Note how TSAs help: a glance through the forward function exposes the functionality of the module

from typing import NamedTuple, List
import copy
import io
import json
import logging
import math
import pathlib
import re
import tarfile

import numpy as np
import torch
from torch.nn import Parameter

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.from_params import FromParams


import sys
sys.path.append('../')
from tsalib import dim_vars, warp
B, T, D, H = dim_vars('Batch SeqLength EmbedDim NumHeads')

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

_ACTIVATION_FUNCTIONS = {
        'relu': torch.nn.ReLU,
        'swish': swish,
        'gelu': gelu
}

class LayerNorm(torch.nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(n_state))
        self.b = torch.nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b

class Conv1D(torch.nn.Module):
    def __init__(self, nf: int, rf: int, nx: int) -> None:
        super().__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            torch.nn.init.normal_(w, std=0.02)
            self.w: (nx, nf) = Parameter(w)
            self.b: (nf,) = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x: (B, T, self.nx)) -> torch.Tensor:
        if self.rf == 1:
            size_out: (B, T, self.nf) = x.size()[:-1] + (self.nf,)
            x1: (B*T, self.nx) = x.view(-1, x.size(-1))
            x: (B*T, self.nf) = torch.addmm(self.b, x1, self.w)
            x: (B, T, self.nf) = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class Attention(torch.nn.Module):
    def __init__(self,
                 nx: int,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False) -> None:
        super().__init__()
        self.nx = nx
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.num_heads == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.num_heads
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = torch.nn.Dropout(config.attention_dropout_probability)
        self.resid_dropout = torch.nn.Dropout(config.residual_dropout_probability)

    def _attn(self, q: (B,H,T,D), k: (B,H,D,T), v: (B,H,T,D)) -> torch.Tensor:
        w: (B,H,T,T) = torch.matmul(q, k) #similarities
        if self.scale:
            w = w / math.sqrt(v.size(-1)) #scaled similarities

        #adding positional encodings?
        w: (B,H,T,T) = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w: (B,H,T,T) = torch.nn.Softmax(dim=-1)(w)
        w: (B,H,T,T) = self.attn_dropout(w)
        res: (B,H,T,D) = torch.matmul(w, v)
        return res

    def merge_heads_old(self, x: (B,H,T,D)):
        # pylint: disable=no-self-use
        x: (B,T,H,D) = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        res: (B,T,H*D) = x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states
        return res

    def merge_heads(self, x: (B,H,T,D)):
        # pylint: disable=no-self-use
        res = warp(x, 'bhtd -> bthd -> b,t,h*d', 'pcv') #permute, then contiguous, then view transforms
        return res

    def split_heads(self, x: (B, T, D), k: bool = False):
        H = self.n_head
        new_x_shape = x.size()[:-1] + (H, x.size(-1) // H)
        x: (B, T, H, D//H) = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            ret: (B, H, D // H, T) = x.permute(0, 2, 3, 1)
        else:
            ret: (B, H, T, D // H) = x.permute(0, 2, 1, 3)

        return ret

    def forward(self, x: (B, T, self.nx)) -> torch.Tensor:
        D = self.split_size
        H = self.n_head
        
        x: (B, T, 3*D) = self.c_attn(x)
        query: (B, T, D); key: (B, T, D); value: (B, T, D)
        query, key, value = x.split(D, dim=2)

        query: (B, H, T, D // H) = self.split_heads(query)
        key: (B, H, D//H, T) = self.split_heads(key, k=True)
        value: (B, H, T, D // H) = self.split_heads(value)
        a: (B,H,T,D//H) = self._attn(query, key, value)
        a: (B,T,D) = self.merge_heads(a)
        a: (B,T,D) = self.c_proj(a)
        a: (B,T,D) = self.resid_dropout(a)
        return a


class MLP(torch.nn.Module):
    def __init__(self, n_state: int, config: TransformerConfig) -> None:  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = Conv1D(n_state, 1, config.embedding_dim)
        self.c_proj = Conv1D(config.embedding_dim, 1, n_state)
        self.act = _ACTIVATION_FUNCTIONS[config.activation_function]
        self.dropout = torch.nn.Dropout(config.residual_dropout_probability)

    def forward(self, x: (B,T,D)) -> torch.Tensor:
        h: (B,T,4*D) = self.act(self.c_fc(x))
        h2: (B, T, D) = self.c_proj(h)
        return self.dropout(h2)


class Block(torch.nn.Module):
    def __init__(self,
                 n_ctx: int,
                 config: TransformerConfig,
                 scale: bool = False) -> None:
        super().__init__()
        nx = config.embedding_dim
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x: (B,T,D)) -> torch.Tensor:
        a: (B,T,D) = self.attn(x)
        n: (B,T,D) = self.ln_1(x + a)
        m: (B, T, D) = self.mlp(n)
        h: (B,T,D) = self.ln_2(n + m)
        return h

class OpenaiTransformer(torch.nn.Module, FromParams):
    """
    Openai transformer, as per https://blog.openai.com/language-unsupervised/.
    Default parameters are the ones for their pretrained model.
    Parameters
    ----------
    vocab_size: ``int`` (optional, default: 40478)
        The size of the vocabulary (number of byte pair embeddings)
        excluding the n_special embeddings (if any), and the positional embeddings.
    n_ctx: ``int`` (optional, default: 512)
        The number of positional encodings to use for evaluation.
    embedding_dim: ``int`` (optional, default: 768)
        The dimension of the output embeddings.
    num_heads: ``int`` (optional, default: 12)
        How many "heads" the attention has.
    num_layers: ``int`` (optional, default: 12)
        How many layers of "blocks" the transformer has.
    embedding_dropout_probability: ``float`` (optional, default: 0.1)
        Dropout for the embedding.
    attention_dropout_probability: ``float`` (optional, default: 0.1)
        Dropout for attention.
    residual_dropout_probability: ``float`` (optional, default: 0.1)
        Dropout for residual
    activation_function: ``str`` (optional, default: ``'gelu'``)
        Activation function for the multi-layer perceptron.
    model_path: ``str`` (optional, default: ``None``)
        A tar.gz file containing serialized model weights. If supplied,
        the weights will be loaded from that file.
    requires_grad: ``bool`` (optional, default: ``False``)
        If true, the transformer will be fine-tuneable.
    n_special: ``int`` (optional, default: ``-1``)
        The number of special tokens added to the byte pair vocabulary
        (via ``OpenaiTransformerBytePairIndexer``).
    """
    def __init__(self,
                 vocab_size: int = 40478,
                 n_ctx: int = 512,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 embedding_dropout_probability: float = 0.1,
                 attention_dropout_probability: float = 0.1,
                 residual_dropout_probability: float = 0.1,
                 activation_function: str = 'gelu',
                 model_path: str = None,
                 requires_grad: bool = False,
                 n_special: int = -1) -> None:
        super().__init__()

        config = TransformerConfig(
                embedding_dim,
                num_heads,
                embedding_dropout_probability,
                attention_dropout_probability,
                residual_dropout_probability,
                activation_function,
        )

        # the embedding size is vocab_size + n_special embeddings + n_ctx
        embedding_size = vocab_size + max(n_special, 0) + n_ctx
        self.vocab_size = embedding_size
        self.n_ctx = n_ctx
        self.n_special = n_special

        self.num_output_layers = 1 + num_layers

        self.embed = torch.nn.Embedding(embedding_size, embedding_dim)
        self.drop = torch.nn.Dropout(embedding_dropout_probability)

        block = Block(n_ctx, config, scale=True)
        self.h = torch.nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        self.decoder = torch.nn.Linear(embedding_dim, embedding_size, bias=False)
        self.decoder.weight = self.embed.weight  # Tied weights
        # To reproduce the noise_shape parameter of TF implementation

        torch.nn.init.normal_(self.embed.weight, std=0.02)

        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

        if model_path:
            self.load_weights(model_path, n_special=n_special, n_ctx=n_ctx)

    def forward(self, x: (B,T)) -> List[B, T, D]:
        # x is (batch_size, sequence_length) tensor of byte-pair ids

        # e is (batch_size, sequence_length, 2, embedding_dim) tensor of embeddings
        e: (B, T, 2, D) = self.embed(x)

        h: (B, T, D) = e.sum(dim=2)

        all_layers: List[B,T,D] = [h]
        for block in self.h:
            h = block(h)
            all_layers.append(h)

        return all_layers