#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=16000
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#--gpus-per-node=3090:1
#SBATCH --gres=gpu:3090:1
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using job and first node values

# Purpose: To iterate over a few different random seeds and create some 
#          base models for the SWAG event/blast classifier.
# Author: Ben Baker (UUSS) distributed under the MIT license.
# Usage: sbatch slurmBase.sh

# Set some directory information
PYTHON_DIR=~/software/pkg/miniforge3/envs/atc/bin

${PYTHON_DIR}/python test_apply_swag_pickers_nodb.py