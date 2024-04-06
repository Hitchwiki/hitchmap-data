#!/bin/sh
#SBATCH -A herbrich
#SBATCH --job-name=feedback2code-myjob
#SBATCH --output ./log/slurm-%j.out
#SBATCH --partition sorcery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gpus=0
#SBATCH --time=10:00:00


python -u ./gp2.py > ./gp2.out