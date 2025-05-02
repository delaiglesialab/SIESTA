#!/bin/bash
#SBATCH --job-name=feature_selection     
#SBATCH --output=sfs_feature_selection_%j.log 
#SBATCH --error=sfs_feature_selection_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=360G 
#SBATCH --time=24:00:00
#SBATCH --partition=ckpt-g2

python3 FeatureSelection.py