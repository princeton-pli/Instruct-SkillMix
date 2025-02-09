#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --job-name=evaluate_lm-evaluation-harness
#SBATCH -t 00:30:00

# 50 minutes for Llama-2-7B on 1 H100 GPU / 10 minutes on 4 H100 GPUs

export PROJECT_DIR=

# adjust for different experiments
model_name=llama2_7b_base                                         # llama2_13b_base, mistral_7b_v0.2_base, llama3_8b_base
k=2                                                               # 1, 3
NUM_TRAIN_EXAMPLES=1000                                           # 2000, 4000
MODEL_STR=${model_name}_sda_k${k}_${NUM_TRAIN_EXAMPLES}           # ${model_name}_sdd_k${k}_${NUM_TRAIN_EXAMPLES}
DATA_PATH=${PROJECT_DIR}/datasets/ism_sda_k${k}_${NUM_TRAIN_EXAMPLES}.json
HELDOUT_DATA_NAME=ism_sda_k${k}_heldout                           # ism_sdd_k${k}_heldout
n_epochs=15
epoch=1                                                           # selected based on heldout performance

# adjustable hyperparameters
LR=0.00002
TRUE_BS=64

# load parameters, environment variables
source ./src/slurm/template.sh

MODEL_STR_EXTENDED=${MODEL_STR}_lr_${LR}_bs_${TRUE_BS}_epoch_${epoch}_of_${N_EPOCHS}

## lm-evaluation-harness
accelerate launch -m lm_eval \
    --model checkpoints \
    --model_args pretrained=${CHECKPOINT_PATH}/epoch_${epoch} \
    --output_path ${EVAL_PATH}/lm-evaluation-harness/model_outputs/${MODEL_STR_EXTENDED} \
    --tasks mmlu,truthfulqa,arc_challenge,winogrande,piqa,gsm8k \
    --batch_size auto