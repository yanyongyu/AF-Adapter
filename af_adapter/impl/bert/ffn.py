import inspect
from functools import wraps
from typing import Dict, Type, TypeVar, Callable, Optional, Protocol, cast

import torch
import torch.nn as nn
from nlppets.general import MonkeyPatch
from nlppets.torch import concat_linear
from transformers.models.bert import BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertOutput as BaseOutput
from transformers.models.bert.modeling_bert import BertIntermediate as BaseIntermediate

MT = TypeVar("MT", bound=Type[BertPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class BertIntermediate(nn.Module):
    enhancements: Dict[str, int]

    # ported from original BertIntermediate
    dense: nn.Linear
    intermediate_act_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, config: Config):
        BaseIntermediate.__init__(cast(BaseIntermediate, self), config)

        dtype = self.dense.weight.dtype
        device = self.dense.weight.device

        self.enhancements = config.domain_ffn_enhance
        for name, size in config.domain_ffn_enhance.items():
            setattr(
                self,
                name,
                nn.Linear(config.hidden_size, size, device=device, dtype=dtype),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, H] -> [B, L, I + E]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    enhancements: Dict[str, int]

    # ported from original BertOutput
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(self, config: Config):
        BaseOutput.__init__(cast(BaseOutput, self), config)

        dtype = self.dense.weight.dtype
        device = self.dense.weight.device

        self.enhancements = config.domain_ffn_enhance
        for name, size in config.domain_ffn_enhance.items():
            setattr(
                self,
                name,
                nn.Linear(size, config.hidden_size, device=device, dtype=dtype),
            )

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, I + E] -> [B, L, H]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BERT model to apply feed-forward network domain enhancement.

    Args:
        model (Type[BertPreTrainedModel]): Original BERT model class.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        Type[BertPreTrainedModel]: Patched model class
    """

    origin_init = model.__init__
    bert_module = inspect.getmodule(model)

    def patched_init(
        self: BertPreTrainedModel, config: PretrainedConfig, *args, **kwargs
    ):
        # patch config if new enhancement provided
        if domain_ffn_enhance is not None:
            config.domain_ffn_enhance = domain_ffn_enhance

        config_with_enhance = cast(Config, config)

        with MonkeyPatch.context() as m:
            # if domain enhance, replace modules
            if config_with_enhance.domain_ffn_enhance:
                m.setattr(bert_module, "BertIntermediate", BertIntermediate)
                m.setattr(bert_module, "BertOutput", BertOutput)

            origin_init(self, config, *args, **kwargs)

    new_model = type(
        f"{model.__name__}_EnhanceFFN", (model,), {"__init__": patched_init}
    )
    return wraps(model, updated=())(new_model)  # type: ignore
