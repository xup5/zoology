"""
Attention approximation, modified from Based.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe
from einops import rearrange
from typing import Optional, Tuple
from pydantic import validate_call

from zoology.utils import import_from_str

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaRotaryEmbedding
except:
    print(f"Failed to import LlamaRotaryEmbedding... ")

try:
    import sys
    sys.path.append('/var/cr05_data/sim_data/code/based/')
    from csrc import causal_dot_product  # linear attention cuda kernel
    print(f"Succesfully imported the causal dot product kernel... ")
except:
    causal_dot_product = None
    print(f"Failed to import the causal dot product kernel... ")

class AttenApprox(nn.Module):
    
    @validate_call
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_heads: int = 12,
        # feature_name: "str" = "taylor_exp",
        # feature_kwargs: dict = {},
        eps: float = 1e-12,
        causal: bool = True,
        apply_rotary: bool = False,
        rope_theta: int=10000.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max

        # linear attention 
        # self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_key_value_groups = self.num_heads // self.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.causal=causal
        # feature_map_kwargs = {
        #     'input_dim': self.feature_dim,
        #     'head_dim_idx': -1,
        #     'temp': 1.,
        #     'eps': 1e-12,
        #     **feature_kwargs
        # }
        # self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

        # parameters
        self.apply_rotary = apply_rotary
        self.rope_theta = rope_theta
        self.q_shape = [self.num_heads, self.feature_dim]
        self.k_shape = [self.num_heads, self.feature_dim]
        self.v_shape = [self.num_heads, self.head_dim]
        if self.apply_rotary:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.feature_dim,
                max_position_embeddings=self.l_max,
                base=self.rope_theta,
            )

    def process_qkv(
        self, 
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        """
        Get Q, K, V tensors from hidden_states, e.g., by applying projections, 
        positional embeddings, KV cache
        -> Follow the original LlamaAttention API
        """
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        
        # Following HF Llama source code to get (b, h, l, d)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)
        
        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        # Apply rotary embeddings
        if position_ids is None:
            position_ids = torch.arange(
                kv_seq_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand((b, kv_seq_len))
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # KV cache
        if past_key_value is not None:
            # Reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            
        past_key_value = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        filters: torch.Tensor=None, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        # b: batch size
        # l: sequence length 
        # d: hidden dimension (d_model)
        b, l, d = hidden_states.size()
        if self.apply_rotary:
            assert d == self.d_model, f'Hidden_states.shape should be size {(b, l, d)} but is shape {hidden_states.shape}'
            q, k, v, kv_seq_len = self.process_qkv(hidden_states, past_key_value, position_ids, use_cache)
        else:
            q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
            q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
            k = k.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
            v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Linear attention
        # q, k = self.feature_map(q), self.feature_map(k)
        
        # Attention approximation
        n = torch.arange(1, q.shape[2]+1, device=q.device, dtype=q.dtype).unsqueeze(-1)
        mean_keys = torch.cumsum(k, dim=2)
        mean_keys = mean_keys/n
        center_keys = k - mean_keys
        mean_values = torch.cumsum(v, dim=2)
        mean_values = mean_values/n
        center_values = v - mean_values
        qK = torch.einsum("bhqi,bhpi->bhqp", q, center_keys) # [batch_size, num_heads, querylength, querylength]
        
        # mask the upper triangular part of qK
        qK = qK.view(-1, qK.shape[2], qK.shape[3]) # [batch_size*num_heads, querylength, querylength]
        qK = torch.tril(qK)
        qK = qK.view(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
        
        qK_squared = torch.cumsum(qK**2/(2*torch.tensor(self.head_dim, device=qK.device, dtype=qK.dtype)), dim=2)
        qK_squared = torch.sum(qK_squared, dim=3)
        
        denominator = n.squeeze() + qK_squared
        
        qKV = torch.einsum("bhqp,bhpi->bhqi", qK, center_values)/torch.sqrt(torch.tensor(self.head_dim, device=qK.device, dtype=qK.dtype))
        
        y = mean_values + qKV/denominator.unsqueeze(-1)

                    

        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.proj_o(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype) 
    

# need to change this.
    def state_size(self, sequence_length: int=2048):
        return (
            1
            # self.num_heads * self.head_dim * self.feature_map.expanded_size() + 
            # self.num_heads * self.feature_map.expanded_size()
        )