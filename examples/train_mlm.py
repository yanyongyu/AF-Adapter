import os
import time
import contextlib
from pathlib import Path
from typing import Tuple, Union

import torch
import evaluate
import torch.distributed
from datasets import Split, Dataset
from nlppets.datasets import RawTextDatasetBuilder
from nlppets.transformers.trainer import train_mlm
from transformers.training_args import OptimizerNames
from nlppets.transformers.tokenize import ChineseWWMTokenizer
from nlppets.torch import (
    combine_model,
    count_parameters,
    count_trainable_parameters,
    calculate_gradient_accumulation,
)
from transformers import (
    WEIGHTS_NAME,
    TrainerState,
    BertTokenizer,
    EvalPrediction,
    TrainerControl,
    BertForMaskedLM,
    TrainerCallback,
    IntervalStrategy,
    TrainingArguments,
    BertForPreTraining,
)

from af_adapter.impl.bert import (
    domain_enhance_att,
    domain_enhance_ffn,
    freeze_original_tensors,
)

# init distribution first to calculate gradient accumulation
if os.getenv("LOCAL_RANK"):
    print("Initializing distribution...")
    torch.distributed.init_process_group(backend="nccl")
    print("Distribution initialized.")

BASE_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
ENHANCEMENTS = ("medical",)
ATT_SIZE = int(os.getenv("ATT_SIZE", "1"))
FFN_SIZE = int(os.getenv("FFN_SIZE", "1024"))
FP16 = bool(os.getenv("FP16", ""))

MODEL_NAME = os.getenv("MODEL_NAME", str(int(time.time())))
OUTPUT_DIR = f"./data/{MODEL_NAME}/output/"
LOG_DIR = f"./data/{MODEL_NAME}/log/"
WANDB_DIR = f"./data/{MODEL_NAME}/wandb/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(WANDB_DIR).mkdir(parents=True, exist_ok=True)

DATASETS_ROOT = Path(os.getenv("DATASETS_ROOT", "./data/datasets/"))
# ! change the datasets dirs and files to your own
# dirs: a list of directories containing text files
# files: a list of text files
DATASETS = {
    "dirs": (DATASETS_ROOT / "dirs",),
    "files": (DATASETS_ROOT / "texts.txt",),
}

SAVE_STEPS = 1000
# SAVE_TOTAL_LIMIT = 10
EVAL_STEPS = 1000

MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "100000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_SEQ_LEN = 512
GRADIENT_ACCUMULATION_STEPS = calculate_gradient_accumulation(BATCH_SIZE, 512)
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.01
WARMUP_STEPS = MAX_TRAIN_STEPS // 100

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to=["all"],
    run_name=MODEL_NAME,
    do_train=True,
    do_eval=True,
    logging_first_step=True,
    logging_dir=LOG_DIR,
    save_steps=SAVE_STEPS,
    # save_total_limit=SAVE_TOTAL_LIMIT,
    max_steps=MAX_TRAIN_STEPS,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=EVAL_STEPS,
    optim=OptimizerNames.ADAMW_TORCH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    fp16=FP16,
)

print("Training mode:", args.parallel_mode, "World size:", args.world_size)


def preprocess_logits(
    logits: Union[torch.Tensor, Tuple[torch.Tensor, ...]], labels: torch.Tensor
):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


print("Loading metric...")
with args.main_process_first(local=False, desc="loading metric"):
    metric = evaluate.load("accuracy")


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds, labels = eval_preds
    labels = labels.reshape(-1)  # type: ignore
    preds = preds.reshape(-1)  # type: ignore
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)  # type: ignore


print("Loading tokenizer...")
with args.main_process_first(local=False, desc="loading tokenizer"):
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)

print("Loading model...")
# modify the model to add enhancements
model_cls = domain_enhance_ffn(
    BertForMaskedLM, {name: FFN_SIZE for name in ENHANCEMENTS}
)
model_cls = domain_enhance_att(model_cls, {name: ATT_SIZE for name in ENHANCEMENTS})

with args.main_process_first(local=False, desc="loading model"):
    model: BertForMaskedLM = model_cls.from_pretrained(BASE_MODEL_NAME)  # type: ignore

# freeze original tensors
model = freeze_original_tensors(model, ENHANCEMENTS)

params = count_parameters(model, deepspeed=True)
trainable = count_trainable_parameters(model, deepspeed=True)
print(
    "Parameters:",
    params,
    "Trainable:",
    trainable,
    "Trainable/Parameters:",
    f"{trainable/params:.2%}",
)

print("Loading dataset...")
dataset_builder = RawTextDatasetBuilder(**DATASETS)
with args.main_process_first(local=False, desc="generating dataset"):
    dataset_builder.download_and_prepare()
    dataset: Dataset = dataset_builder.as_dataset(split=Split("train"))  # type: ignore

processor = ChineseWWMTokenizer(tokenizer, MAX_SEQ_LEN)
with args.main_process_first(local=False, desc="grouping texts together"):
    dataset = dataset.map(
        processor.batched_tokenize_group_texts,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Grouping texts in chunks of {MAX_SEQ_LEN}",
    )

dataset_dict = dataset.train_test_split(test_size=0.05)
eval_dataset = dataset_dict["test"]
train_dataset = dataset_dict["train"]

print("Train dataset:", train_dataset)
print("Eval dataset:", eval_dataset)


old_model = BertForPreTraining.from_pretrained(BASE_MODEL_NAME, device_map="cpu").state_dict()  # type: ignore


def patch_output_model(dir: str):
    new_model = (
        domain_enhance_att(domain_enhance_ffn(BertForMaskedLM))
        .from_pretrained(dir)
        .state_dict()  # type: ignore
    )
    new_state = combine_model(old_model, new_model)
    torch.save(new_state, Path(dir) / WEIGHTS_NAME)


class PatchOutputModel(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        patch_output_model(str(checkpoint_dir))


if __name__ == "__main__":
    print("Start training...")

    if (project := os.getenv("WANDB_PROJECT", "")) and args.local_process_index == 0:
        import wandb

        run = wandb.init(
            project=project,
            name=args.run_name,
            dir=WANDB_DIR,
            tags=["AF-Adapter", "Bert", "medical", "transfer"],
            resume=True,
        )
    else:
        run = contextlib.nullcontext()

    with run:  # type: ignore
        train_mlm(
            model=model,
            tokenizer=tokenizer,
            training_args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            wwm=True,
            compute_metrics=compute_metrics,
            callbacks=[PatchOutputModel()],
            preprocess_logits=preprocess_logits,
        )
        patch_output_model(OUTPUT_DIR)
