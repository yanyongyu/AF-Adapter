import math
import inspect
from functools import wraps
from typing import Any, Dict, Type, Tuple, Literal, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from nlppets.general import MonkeyPatch
from nlppets.torch import concat_linear
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfOutput as BaseSelfOutput
from transformers.models.bert.modeling_bert import (
    BertSelfAttention as BaseSelfAttention,
)

MT = TypeVar("MT", bound=Type[BertPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    num_attention_heads: int
    domain_att_enhance: Dict[str, int]
    """domain pre-training enhancements."""


class BertSelfAttention(nn.Module):
    enhancements: Dict[str, int]
    additional_heads: int

    # ported from original BertSelfAttention
    num_attention_heads: int
    attention_head_size: int
    all_head_size: int
    query: nn.Linear
    key: nn.Linear
    value: nn.Linear
    dropout: nn.Dropout
    position_embedding_type: str
    max_position_embeddings: int
    distance_embedding: nn.Embedding
    is_decoder: bool

    def __init__(self, config: Config):
        BaseSelfAttention.__init__(cast(BaseSelfAttention, self), config)

        dtype = self.key.weight.dtype
        device = self.key.weight.device

        self.enhancements = config.domain_att_enhance
        self.additional_heads = sum(config.domain_att_enhance.values())
        for name, size in config.domain_att_enhance.items():
            setattr(
                self,
                f"{name}_query",
                nn.Linear(
                    config.hidden_size,
                    self.attention_head_size * size,
                    device=device,
                    dtype=dtype,
                ),
            )
            setattr(
                self,
                f"{name}_key",
                nn.Linear(
                    config.hidden_size,
                    self.attention_head_size * size,
                    device=device,
                    dtype=dtype,
                ),
            )
            setattr(
                self,
                f"{name}_value",
                nn.Linear(
                    config.hidden_size,
                    self.attention_head_size * size,
                    device=device,
                    dtype=dtype,
                ),
            )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, L, HN, HS]
        new_x_shape = x.size()[:-1] + (
            x.size()[-1]
            // self.attention_head_size,  # calculate head num automatically
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        # [B, L, HN, HS] -> [B, HN, L, HS]
        return x.permute(0, 2, 1, 3)

    def _patch_for_domain_enhance(
        self, hidden_states: torch.Tensor, type: Literal["query", "key", "value"]
    ) -> torch.Tensor:
        return concat_linear(
            getattr(self, type),
            *(getattr(self, f"{name}_{type}") for name in self.enhancements),
        )(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Any, ...]:
        # [B, L, H] -> [B, L, H]
        mixed_query_layer = self._patch_for_domain_enhance(hidden_states, "query")

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k, v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(encoder_hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(encoder_hidden_states, "value")
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "value")
            )
            # concat over the length dim
            key_layer = torch.concat((past_key_value[0], key_layer), dim=2)
            value_layer = torch.concat((past_key_value[1], value_layer), dim=2)
        else:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "value")
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [B, HN + E, L, HS], [B, HN + E, L, HS] -> [B, HN + E, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type in {"relative_key", "relative_key_query"}:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs: torch.Tensor = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # [B, HN + E, L, L], [B, HN + E, L, HS] -> [B, HN + E, L, HS]
        context_layer = torch.matmul(attention_probs, value_layer)

        # [B, HN + E, L, HS] -> [B, L, H + E * HS]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size + self.additional_heads * self.attention_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    enhancements: Dict[str, int]
    hidden_size: int
    attention_head_size: int

    # ported from original BertSelfOutput
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(self, config: Config):
        BaseSelfOutput.__init__(cast(BaseSelfOutput, self), config)

        dtype = self.dense.weight.dtype
        device = self.dense.weight.device

        self.enhancements = config.domain_att_enhance
        self.hidden_size = config.hidden_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        for name, size in config.domain_att_enhance.items():
            setattr(
                self,
                name,
                nn.Linear(
                    self.attention_head_size * size,
                    config.hidden_size,
                    device=device,
                    dtype=dtype,
                ),
            )

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # patched for domain enhancement
        # [B, L, H + E * HS] -> [B, L, H]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def domain_enhance_att(
    model: MT, domain_att_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BERT model to apply self attention domain enhancement.

    Args:
        model (BertPreTrainedModel | Type[BertPreTrainedModel]): Original BERT model class.
        domain_att_enhance (Optional[Dict[str, int]]): Domain enhancements.
            If None is provided, will read from existing configs.

    Returns:
        BertPreTrainedModel | Type[BertPreTrainedModel]: Patched model class
    """

    origin_init = model.__init__
    bert_module = inspect.getmodule(model)

    def patched_init(
        self: BertPreTrainedModel, config: PretrainedConfig, *args, **kwargs
    ):
        # patch config if new enhancement provided
        if domain_att_enhance is not None:
            config.domain_att_enhance = domain_att_enhance

        config_with_enhance = cast(Config, config)

        with MonkeyPatch.context() as m:
            # if domain enhance, replace modules
            if config_with_enhance.domain_att_enhance:
                m.setattr(bert_module, "BertSelfAttention", BertSelfAttention)
                m.setattr(bert_module, "BertSelfOutput", BertSelfOutput)

            origin_init(self, config, *args, **kwargs)

    new_model = type(
        f"{model.__name__}_EnhanceATT", (model,), {"__init__": patched_init}
    )
    return wraps(model, updated=())(new_model)  # type: ignore
