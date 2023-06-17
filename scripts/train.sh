#!/bin/bash

echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES

if [ -n $SLURMD_NODENAME ]; then
  echo "Node: " $SLURMD_NODENAME
fi

if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  echo "RANK: " $RANK "/" $WORLD_SIZE
  echo "LOCAL_RANK: " $SLURMD_NODENAME $LOCAL_RANK "/" $SLURM_NTASKS_PER_NODE
fi

if [ -n $CONDA_ENV ]; then
  echo "Using $(conda --version), $(conda run -n $CONDA_ENV python --version)"
  env_python=$(conda run -n $CONDA_ENV which python)
else
  env_python=$(which python)
  echo "Using $($env_python --version)"
fi

# wandb configuration, canbe removed if not using wandb
conda run -n $CONDA_ENV wandb login "<your-api-key-hear>"
conda run -n $CONDA_ENV wandb status
export WANDB_PROJECT=bert-medical-af-adapter

export MODEL_NAME=${MODEL_NAME:-"bert-medical-af-adapter-v1"}

export BATCH_SIZE=${BATCH_SIZE:-64}
export FP16=${FP16:-"true"}

TRAIN_FILE=${TRAIN_FILE:-"./examples/train_mlm.py"}
$env_python -u $TRAIN_FILE
