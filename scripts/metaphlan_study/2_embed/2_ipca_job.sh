#!/bin/bash
#SBATCH --partition=bwh_comppath
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --job-name=ipca
#SBATCH --output=logs/ipca_%j.out
#SBATCH --error=logs/ipca_%j.err


python 2_ipca_dimreduce.py
echo "Done."
