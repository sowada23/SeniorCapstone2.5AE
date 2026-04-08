#!/bin/bash
#SBATCH --job-name=UAD      
#SBATCH --output=output_%j.txt     
#SBATCH --error=error_%j.txt       
#SBATCH --time=1:00:00            
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=f3
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --mail-user=sowada23@earlham.edu  

set -e

cd /cluster/home/sowada23/MedAI/UAD/AutoEncoder/2.5D
source .venv/bin/activate

echo "Starting training..."
time python scripts/train.py --config configs/default.yaml

echo "Starting testing..."
time python scripts/test.py --config configs/default.yaml --output-root ./Output