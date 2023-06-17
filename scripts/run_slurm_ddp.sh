#!/bin/bash
#SBATCH --job-name=bert-medical-af-adapter
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=2

. ./run.sh
