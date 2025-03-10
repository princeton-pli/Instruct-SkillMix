#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --job-name=evaluate_AlpacaEval
#SBATCH -t 01:00:00

# 40 minutes for Llama-2-7B on 1 H100 GPU

export PROJECT_DIR=

# adjust for different experiments
model_name=llama2_7b_base                                         # llama2_13b_base, mistral_7b_v0.2_base, llama3_8b_base
k=2                                                               # 1, 3
NUM_TRAIN_EXAMPLES=1000                                           # 2000, 4000
MODEL_STR=${model_name}_sda_k${k}_${NUM_TRAIN_EXAMPLES}           # ${model_name}_sdd_k${k}_${NUM_TRAIN_EXAMPLES}
n_epochs=15
epoch=1                                                           # selected based on heldout performance

# adjustable hyperparameters
LR=0.00002
TRUE_BS=64

# load parameters, environment variables
source ./src/slurm/template.sh

MODEL_STR_EXTENDED=${MODEL_STR}_lr_${LR}_bs_${TRUE_BS}_epoch_${epoch}_of_${N_EPOCHS}
EVAL_CONFIG_DIR=${EVAL_CONFIG_PATH}/sft_models/${MODEL_STR_EXTENDED}
    
## AlpacaEval
alpaca_eval evaluate_from_model \
    --model_configs ${EVAL_CONFIG_DIR} \
    --output_path ${EVAL_PATH}/alpaca_eval_2/model_outputs/${MODEL_STR_EXTENDED}