import torch.nn as nn
import json
import torch
from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN
from p_choose import PChooseLayer
from attention import MarianAttention
from loss import isNAN_isINF

class MarianDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.embed_dim = self.config['d_model']

        self.self_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=self.config['decoder_attention_heads'],
            dropout=self.config['attention_dropout'],
            is_decoder=True,
        )
        
        self.dropout = self.config['dropout']
        self.activation_fn = ACT2FN[self.config['activation_function']]
        self.activation_dropout = self.config['activation_dropout']

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MarianAttention(
            self.embed_dim,
            self.config['decoder_attention_heads'],
            dropout=self.config['attention_dropout'],
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ### add p_choice layer
        self.p_choose_layer = PChooseLayer(self.config['d_model'], self.config['decoder_attention_heads'], energy_bias_value=-0.5, monotonic_temperature=0.2, num_monotonic_energy_layers=4, pre_decision_ratio=2, dtype=torch.float)
        # self.p_choose_layer = PChooseLayer(self.config['d_model'], self.config['decoder_attention_heads'], energy_bias_value=-1.0, monotonic_temperature=0.2, num_monotonic_energy_layers=4, pre_decision_ratio=2, dtype=torch.float)
        self.fc1 = nn.Linear(self.embed_dim, self.config['decoder_ffn_dim'])
        self.fc2 = nn.Linear(self.config['decoder_ffn_dim'], self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.Tensor] = None,
    #     layer_head_mask: Optional[torch.Tensor] = None,
    #     cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: Optional[bool] = False,
    #     use_cache: Optional[bool] = False,
    #     is_training : bool = False
    # ):
    #     """
    #     Args:
    #         hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
    #         attention_mask (:obj:`torch.FloatTensor`): attention mask of size
    #             `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
    #         encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
    #         encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
    #             `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
    #         layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
    #             `(encoder_attention_heads,)`.
    #         cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
    #             size `(decoder_attention_heads,)`.
    #         past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
    #         output_attentions (:obj:`bool`, `optional`):
    #             Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
    #             returned tensors for more detail.
    #     """
    #     residual = hidden_states

    #     # Self Attention
    #     # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    #     self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    #     # add present self-attn cache to positions 1,2 of present_key_value tuple
    #     hidden_states, self_attn_weights, present_key_value = self.self_attn(
    #         hidden_states=hidden_states,
    #         past_key_value=self_attn_past_key_value,
    #         attention_mask=attention_mask,
    #         layer_head_mask=layer_head_mask,
    #         output_attentions=output_attentions,
    #     )
    #     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    #     hidden_states = residual + hidden_states
    #     hidden_states = self.self_attn_layer_norm(hidden_states)

    #     p_choose = self.p_choose_layer(hidden_states, encoder_hidden_states)
    #     # print('p_choose:', p_choose)

    #     # Cross-Attention Block
    #     cross_attn_present_key_value = None
    #     cross_attn_weights = None
    #     if encoder_hidden_states is not None:
    #         residual = hidden_states

    #         # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
    #         cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
    #         if is_training:
    #             # print('is_training work, go to training_forward')
    #             hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn.training_forward(
    #                 hidden_states=hidden_states,
    #                 p_choose=p_choose,
    #                 key_value_states=encoder_hidden_states,
    #                 attention_mask=encoder_attention_mask,
    #                 layer_head_mask=cross_attn_layer_head_mask,
    #                 past_key_value=cross_attn_past_key_value,
    #                 output_attentions=output_attentions,
    #             )
    #         else:
    #             hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
    #                 hidden_states=hidden_states,
    #                 key_value_states=encoder_hidden_states,
    #                 attention_mask=encoder_attention_mask,
    #                 layer_head_mask=cross_attn_layer_head_mask,
    #                 past_key_value=cross_attn_past_key_value,
    #                 output_attentions=output_attentions,
    #             )
    #         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    #         hidden_states = residual + hidden_states
    #         hidden_states = self.encoder_attn_layer_norm(hidden_states)

    #         # add cross-attn to positions 3,4 of present_key_value tuple
    #         present_key_value = present_key_value + cross_attn_present_key_value

    #     # Fully Connected
    #     residual = hidden_states
    #     hidden_states = self.activation_fn(self.fc1(hidden_states))
    #     hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    #     hidden_states = self.fc2(hidden_states)
    #     hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    #     hidden_states = residual + hidden_states
    #     hidden_states = self.final_layer_norm(hidden_states)

    #     outputs = (hidden_states,)

    #     if output_attentions:
    #         outputs += (self_attn_weights, cross_attn_weights)

    #     if use_cache:
    #         outputs += (present_key_value,)

    #     return outputs, p_choose

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        is_training : bool = False
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if(isNAN_isINF(residual)):
            with open('outputLOG.txt', 'a')as f:
                print('hidden_states before self-attn', file=f)
                print(hidden_states, file=f)
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        if(isNAN_isINF(hidden_states)):
            with open('outputLOG.txt', 'a')as f:
                print('hidden_states after self-attn', file=f)
                print(hidden_states, file=f)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        
        
        # print('p_choose:', p_choose)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            p_choose = self.p_choose_layer(hidden_states, encoder_hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            if is_training:
                # print('is_training work, go to training_forward')
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn.training_forward(
                    hidden_states=hidden_states,
                    p_choose=p_choose,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            else:
                hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(residual)
            

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        if(isNAN_isINF(hidden_states)):
            with open('outputLOG.txt', 'a')as f:
                print('hidden_states after activation function', file=f)
                print(hidden_states, file=f)

        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if(isNAN_isINF(hidden_states)):
            with open('outputLOG.txt', 'a')as f:
                print('hidden_states after final_layer_norm', file=f)
                print(hidden_states, file=f)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, p_choose
