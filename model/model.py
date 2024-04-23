import math
import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tools.config import ModelConfig

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        assert d_model % 2 == 0

        pos = torch.arange(0, max_seq_len)
        exp = torch.arange(0, d_model, step=2) / d_model
        freq = torch.exp(exp * -torch.log(torch.tensor(10000.0)))
        freq = torch.einsum("i, j -> ij", pos, freq)
        pe = torch.zeros((max_seq_len, d_model))
        pe[:, 0::2] = torch.sin(freq)
        pe[:, 1::2] = torch.cos(freq)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[-2]]

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kvheads: int, dropout: float = 0.1):
        super().__init__()
        assert n_heads % n_kvheads == 0
        assert d_model % n_heads == 0

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.n_kvheads = n_kvheads

        self.qkv = nn.Linear(d_model, self.head_dim * (n_heads + 2 * n_kvheads), bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.attn_score_dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask: Optional[torch.Tensor]):
        batch_size, seq_len, _ = x.shape
        assert len(x.shape) == 3
        qkv = self.qkv(x)
        q_heads, k_heads, v_heads = torch.split(qkv, [self.n_heads * self.head_dim, self.n_kvheads * self.head_dim, self.n_kvheads * self.head_dim], dim=-1)
        num_groups = self.n_heads * self.head_dim // (self.n_kvheads * self.head_dim)
        # shape (batch, group, n_kvheads, seqlen, head_dim)
        q_heads = q_heads.reshape(batch_size, num_groups, self.n_kvheads, seq_len, self.head_dim)
        # shape (batch, n_kvheads, seqlen, head_dim)
        k_heads = k_heads.reshape(batch_size, self.n_kvheads, seq_len, self.head_dim)
        v_heads = v_heads.reshape(batch_size, self.n_kvheads, seq_len, self.head_dim)

        # shape (batch, group, n_kvheads, seqlen, seqlen)
        attn_scores = torch.einsum("b g h l d, b h j d -> b g h l j", q_heads, k_heads)
        attn_scores *= self.head_dim ** -0.5
        if mask is not None:
            attn_scores += mask
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.attn_score_dropout(attn_scores)
        # shape (batch, group, n_kvheads, seqlen, head_dim)
        output = torch.einsum("b g h j l, b h l d -> b g h j d", attn_scores, v_heads)
        # shape (batch, seqlen, model_dim)
        output = rearrange(output, "b g h l d -> b l (g h d)")
        return(self.out(output))

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_intermediate)
        self.gate = nn.Linear(d_model, d_intermediate)
        self.down = nn.Linear(d_intermediate, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        up = self.dropout(self.up(x))
        gate = self.dropout(F.gelu(self.gate(x), approximate="tanh"))
        fuse = up * gate
        output = self.dropout(self.down(fuse))
        return output

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kvheads: int, d_intermediate: int, p_dropout: float, eps: float):
        super().__init__()
        self.attn = Attention(d_model, n_heads, n_kvheads, dropout=p_dropout)
        self.ffn = FeedForwardNetwork(d_model, d_intermediate, dropout=p_dropout)
        self.pre_attn_norm = nn.LayerNorm(d_model, eps)
        self.post_attn_norm = nn.LayerNorm(d_model, eps)

    def forward(self, x, mask: Optional[torch.Tensor]):
        residual = x
        x = self.pre_attn_norm(x)
        attn_output= self.attn(x, mask)
        x = attn_output + residual

        residual = x
        x = self.post_attn_norm(x)
        ffn_output = self.ffn(x)
        output = ffn_output + residual

        return output

class DecoderModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.decoder_layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.decoder_layers.append(
                DecoderBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_kvheads,
                    cfg.d_intermediate,
                    cfg.p_dropout,
                    cfg.eps,
                )
            )
        self.pe = PositionalEmbeddings(cfg.d_model, cfg.max_seq_len)
        self.input_embeddings = InputEmbeddings(cfg.vocab_size, cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.pe_dropout = nn.Dropout(p=cfg.p_dropout)
        self.input_embeddings_dropout = nn.Dropout(p=cfg.p_dropout)
        self.final_norm = nn.LayerNorm(cfg.d_model, cfg.eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None):
        b, t = idx.size()
        assert t <= self.cfg.max_seq_len, f"Sequence length is too long of length {t}, max sequence length is {self.cfg.max_seq_len}"

        mask = torch.triu(torch.ones((t, t), device=idx.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        hidden_states = self.input_embeddings(idx)
        hidden_states = self.input_embeddings_dropout(hidden_states)
        hidden_states = self.pe(hidden_states)
        hidden_states = self.pe_dropout(hidden_states)
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, mask)
        hidden_states = self.final_norm(hidden_states)

        if targets is not None:
            # we want to change logits.shape = [b, t, v] into [b*t, v] and targets into [b*t] for cross entropy
            logits = self.output(hidden_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.output(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".pe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    # nanogpt
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer