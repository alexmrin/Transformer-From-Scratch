import math
from typing import Optional
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.config import ModelConfig

class RotaryPositionalEmbeddings(nn.Module):
    """
    https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
    """
    def __init__(self, dim: int, max_seq_len: int, base: int):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Note to self: dim/2 MUST be even
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        # We need to create a cache of indices [0, 1, ..., max_seq_len-1]
        seq_idx = torch.arange(0, max_seq_len, dtype=theta.dtype, device=theta.device)
        # shape [max_seq_len, dim//2]
        freqs = torch.einsum("i, j -> ij", seq_idx, theta).float()
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape should be [batch_size, seq_len, n_heads, head_dim]
        seq_len = x.shape[1]
        rope_cache = self.cache[:seq_len]
        # shape [batch_size, seq_len, n_heads, head_dim//2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        # shape [1, seq_len, 1, head_dim//2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kvheads: int, pos_embeddings: RotaryPositionalEmbeddings, attn_dropout: float=0):
        super().__init__()
        assert n_heads % n_kvheads == 0, "number of kv heads must divide query heads."
        assert d_model % n_heads == 0, "number of heads must divide d_model."

        self.n_kvheads = n_kvheads
        self.head_dim = d_model // n_heads
        self.q_per_head = n_heads // n_kvheads
        self.attn_dropout = attn_dropout
        self.q_proj = nn.Linear(d_model, self.head_dim * n_heads, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kvheads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kvheads, bias=False)

        self.pos_embeddings = pos_embeddings
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Need to query into reshape into [batch_size, seq_len, n_kvheads, q_per_head, head_dim]
        q = q.view(b, s, self.n_kvheads, self.q_per_head, self.head_dim)
        # Need to k, v into reshape into [batch_size, seq_len, n_kvheads, 1, head_dim]
        k = k.view(b, s, self.n_kvheads, 1, self.head_dim)
        v = v.view(b, s, self.n_kvheads, 1, self.head_dim)

        k = k.expand(b,s , self.n_kvheads, self.q_per_head, self.head_dim)
        v = v.expand(b,s , self.n_kvheads, self.q_per_head, self.head_dim)

        # Reshape into [batch_size, seq_len, n_heads, head_dim]
        q = q.reshape(b, s, -1, self.head_dim)
        k = k.reshape(b, s, -1, self.head_dim)
        v = v.reshape(b, s, -1, self.head_dim)

        # Positional Embeddings
        q = self.pos_embeddings(q)
        k = self.pos_embeddings(k)

        # Reshape for flash attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
        )
        # Reshape back to [batch_size, seq_len, n_heads, head_dim] and then [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.out_proj(output)
        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int, ffn_dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_intermediate)
        self.gate = nn.Linear(d_model, d_intermediate)
        self.down = nn.Linear(d_intermediate, d_model)
        self.dropout = nn.Dropout(p=ffn_dropout)

    def forward(self, x):
        up = self.dropout(self.up(x))
        gate = self.dropout(F.gelu(self.gate(x), approximate="tanh"))
        fuse = up * gate
        output = self.dropout(self.down(fuse))
        return output
    
class DecoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kvheads: int,
            d_intermediate: int,
            eps: float,
            attn_dropout: float,
            ffn_dropout: float,
            pos_embeddings: RotaryPositionalEmbeddings,
        ):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kvheads, pos_embeddings, attn_dropout)
        self.ffn = FeedForwardNetwork(d_model, d_intermediate, ffn_dropout=ffn_dropout)
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
        self.pe = RotaryPositionalEmbeddings(dim=cfg.d_model//cfg.n_heads, max_seq_len=cfg.max_seq_len, base=cfg.rotary_base)
        for _ in range(cfg.n_layers):
            self.decoder_layers.append(
                DecoderBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    n_kvheads=cfg.n_kvheads,
                    d_intermediate=cfg.d_intermediate,
                    attn_dropout=cfg.attn_dropout,
                    ffn_dropout=cfg.ffn_dropout,
                    eps=cfg.eps,
                    pos_embeddings=self.pe
                )
            )
        self.input_embeddings = InputEmbeddings(cfg.vocab_size, cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.input_embeddings_dropout = nn.Dropout(p=cfg.emb_dropout)
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