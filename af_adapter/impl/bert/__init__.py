from typing import TypeVar, Iterable

from nlppets.torch import nested_freeze_tensor
from transformers.models.bert import BertPreTrainedModel

from .att import domain_enhance_att as domain_enhance_att
from .ffn import domain_enhance_ffn as domain_enhance_ffn

PATCHED_FFN_ADAPTERS = (
    "encoder.layer.*.intermediate.{domain}.*",
    "encoder.layer.*.output.{domain}.*",
)
PATCHED_ATTENTION_ADAPTERS = (
    "encoder.layer.*.attention.self.{domain}_query.*",
    "encoder.layer.*.attention.self.{domain}_key.*",
    "encoder.layer.*.attention.self.{domain}_value.*",
    "encoder.layer.*.attention.output.{domain}.*",
)
PATCHED_ADAPTERS = PATCHED_FFN_ADAPTERS + PATCHED_ATTENTION_ADAPTERS

M = TypeVar("M", bound=BertPreTrainedModel)


def freeze_original_tensors(model: M, domains: Iterable[str]) -> M:
    target = PATCHED_ADAPTERS

    # if model is a wrapper of a base model, we need to add the base model prefix
    base_model_prefix = model.base_model_prefix
    if hasattr(model, base_model_prefix):
        target = tuple(f"{base_model_prefix}.{x}" for x in target)

    target = tuple(x.format(domain=domain) for x in target for domain in domains)
    return nested_freeze_tensor(model, exclude=target)
