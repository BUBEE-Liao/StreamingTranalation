import torch.nn as nn
import torch
from typing import Optional, Tuple, Union
from loss import isNAN_isINF
from torch.nn import AvgPool1d

class MarianAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # self.k_proj_scratch = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj_scratch = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.q_proj_scratch = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.out_proj_scratch = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.pre_decision_ratio = 2
        self.keys_pooling = AvgPool1d(
            kernel_size=self.pre_decision_ratio,
            stride=self.pre_decision_ratio,
            ceil_mode=True,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            ### for pooling ###
            pooled_keys = self.keys_pooling(key_value_states.transpose(1, 2))
            pooled_keys = pooled_keys.transpose(1, 2)
            ###
            key_states = self._shape(self.k_proj(pooled_keys), -1, bsz)
            value_states = self._shape(self.v_proj(pooled_keys), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        # print('query_states:', query_states.shape)
        # print('key_states:', key_states.shape)
        # print('value_states:', value_states.shape)

        query_states = query_states.view(*proj_shape) #shape:(bsz * self.num_heads, SeqD, dim_head)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        

        src_len = key_states.size(1)
        query_states = torch.clamp(query_states, min=1e-7, max=1-1e-7)
        query_states = torch.nan_to_num(query_states,  nan=0.0)
        key_states = torch.clamp(key_states, min=1e-7, max=1-1e-7)
        key_states = torch.nan_to_num(key_states,  nan=0.0)
        value_states = torch.clamp(value_states, min=1e-7, max=1-1e-7)
        value_states = torch.nan_to_num(value_states,  nan=0.0)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = torch.clamp(attn_weights, min=1e-7, max=1-1e-7)
        if(isNAN_isINF(attn_weights)):
            with open('outputLOG.txt', 'a')as f:
                print('attn_weights after self-attn bmm', file=f)
                print(attn_weights, file=f)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if(isNAN_isINF(attn_weights)):
            with open('outputLOG.txt', 'a')as f:
                print('attn_weights after self-attn softmax', file=f)
                print(attn_weights, file=f)


        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = torch.clamp(attn_probs, min=1e-7, max=1-1e-7)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = torch.clamp(attn_output, min=1e-7, max=1-1e-7)
        if(isNAN_isINF(attn_output)):
            with open('outputLOG.txt', 'a')as f:
                print('attn_weights after self-attn QKV',file=f)
                print(attn_weights, file=f)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = torch.clamp(attn_output, min=1e-7, max=1-1e-7)
        attn_output = torch.nan_to_num(attn_output,  nan=0.0)
        if(isNAN_isINF(attn_output)):
            with open('outputLOG.txt', 'a')as f:
                print('attn_output after self-attn out projection', file=f)
                print(attn_weights, file=f)

        return attn_output, attn_weights_reshaped, past_key_value
        
    def calculate_beta(self, alpha, energy):
        bsz, tgt_len, src_len = alpha.size()
        beta_i = []
        for i in range(tgt_len):
            beta_i.append(energy[:, [i]] * torch.flip(torch.cumsum(torch.flip(alpha[:, [i]] * (1/torch.cumsum(energy[:, [i]], dim=-1)), dims=[2]), dim=-1), dims=[2]))
        return torch.cat(beta_i, dim=1)

    def monotonic_alignment(self, p):
        bsz, tgt_len, src_len = p.size()
        
        p_ext = p.roll(1, [-1]).unsqueeze(-2).expand(-1, -1, src_len, -1).triu(1)

        T = (1 - p_ext).cumprod(-1).triu()

        alpha = [torch.bmm(p[:, [0]], T[:, [0]].squeeze(dim=1))]

        for i in range(1, tgt_len):
            alpha.append(p[:, [i]] * torch.bmm(alpha[i - 1], T[:, [i]].squeeze(dim=1)))
        return torch.cat(alpha, dim=1)
        
    def training_forward(
        self,
        hidden_states: torch.Tensor,
        p_choose,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # if is_cross_attention and past_key_value is not None:
        #     # reuse k,v, cross_attentions
        #     key_states = past_key_value[0]
        #     value_states = past_key_value[1]
        # elif is_cross_attention:
            # cross_attentions
        
        ### for pooling
        pooled_keys = self.keys_pooling(key_value_states.transpose(1, 2))
        pooled_keys = pooled_keys.transpose(1, 2)
        ###
        key_states = self._shape(self.k_proj(pooled_keys), -1, bsz)
        value_states = self._shape(self.v_proj(pooled_keys), -1, bsz)
        # elif past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # else:
        #     # self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        # print('query_states:', query_states.shape)
        # print('key_states:', key_states.shape)
        # print('value_states:', value_states.shape)
        
        p_choose = p_choose.view((bsz * self.num_heads, tgt_len, -1))
        # print('p_choose:', p_choose)
        query_states = query_states.view(*proj_shape) #shape:(bsz * self.num_heads, SeqD, dim_head)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        

        src_len = key_states.size(1)
        energy = torch.bmm(query_states, key_states.transpose(1, 2))
        if(isNAN_isINF(energy)):
            with open('outputLOG.txt', 'a')as f:
                print('energy', file=f)
                print(energy, file=f)

        # print('energy:', energy)
        alpha = self.monotonic_alignment(p_choose)
        if(isNAN_isINF(alpha)):
            with open('outputLOG.txt', 'a')as f:
                print('alpha', file=f)
                print(alpha, file=f)
        # print('alpha:', alpha)
        beta = self.calculate_beta(alpha, energy)
        beta = torch.clamp(beta, min=1e-7, max=1-1e-7)
        beta = torch.nan_to_num(beta,  nan=1.0)
        if(isNAN_isINF(beta)):
            with open('outputLOG.txt', 'a')as f:
                print('beta', file=f)
                print(beta, file=f)
        # print('beta:', beta)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if beta.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {beta.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            beta = beta.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            beta = beta.view(bsz * self.num_heads, tgt_len, src_len)
        ### normalization
        # beta -= beta.min(1, keepdim=True)[0]
        # beta /= beta.max(1, keepdim=True)[0]
        # beta = self.LogSoftmax(beta)
        # z = beta-beta.max(1, keepdim=True)[0]
        beta = nn.functional.softmax(beta, dim=-1)
        if(isNAN_isINF(beta)):
            with open('outputLOG.txt', 'a')as f:
                print('beta after softmax', file=f)
                print(beta, file=f)
        # beta = self.LogSoftmax(beta)
        # with open('outputLOG.txt', 'a')as f:
        #     print('after softmax:\n', beta, file=f)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            beta = layer_head_mask.view(1, -1, 1, 1) * beta.view(bsz, self.num_heads, tgt_len, src_len)
            beta = beta.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = beta.view(bsz, self.num_heads, tgt_len, src_len)
            beta = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(beta, p=self.dropout, training=self.training)
        # print('attn_probs:', attn_probs)
        # print('attn_probs.sahpe:', attn_probs.shape)

        # print('value_states:', value_states)
        # print('value_states.shape:', value_states.shape)
        attn_output = torch.bmm(attn_probs, value_states)
        if(isNAN_isINF(attn_output)):
            print('attn_output after bmm')
        # with open('outputLOG.txt', 'a')as f:
        #     print('attn_output after bmm:\n', attn_output, file=f)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        if(isNAN_isINF(attn_output)):
            print('attn_output after final projection')
        # print('attn_output after projection:', attn_output)
        # print("training_forward done, everything is fine")
        # print('attn_output:',attn_output)
        return attn_output, attn_weights_reshaped, past_key_value