#!/bin/bash
#
# Join stdout and sterr in submit.o{job_id}
# Set the queue and the resources
#

#SBATCH --job-name=exalatex2
#SBATCH --gres=gpu:2
#SBATCH --time=00:02:00
#SBATCH --partition=gpu-cascade
#SBATCH --qos=gpu


echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"
export OMP_NUM_THREADS=2
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

./solution.exe
