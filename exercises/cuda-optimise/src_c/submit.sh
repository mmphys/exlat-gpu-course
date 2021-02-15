#!/bin/bash
#
# opencl-intro
#
# Join stdout and sterr in submit.o{job_id}
# Set the queue and the resources
#

#SBATCH --job-name=Optimise
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu-skylake
#SBATCH --qos=gpu


echo "CUDA_VISIBLE_DEVICES set to ${CUDA_VISIBLE_DEVICES}"

./reconstruct
