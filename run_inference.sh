#!/bin/bash

#SBATCH --job-name=bm_inference
#SBATCH --account=EUHPC_R04_192
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.log

# Load your environments
source .venvln/bin/activate

for arg in "$@"
do
  echo "Running: python3 bm_inference.py \"$arg\""
  python3 bm_inference.py "$arg"
done