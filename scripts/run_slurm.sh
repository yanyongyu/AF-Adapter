#!/bin/bash
#SBATCH --job-name=bert-medical-af-adapter
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

echo "Nodelist: " $SLURM_JOB_NODELIST
echo "Number of nodes: " $SLURM_JOB_NUM_NODES
echo "Number of tasks per node: " $SLURM_NTASKS_PER_NODE

if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
  export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
  echo "WORLD_SIZE: " $WORLD_SIZE

  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
  echo "MASTER_ADDR: " $MASTER_ADDR
  echo "MASTER_PORT: " $MASTER_PORT
fi

srun ./train.sh
