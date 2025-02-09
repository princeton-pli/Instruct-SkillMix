#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --job-name=finetune
#SBATCH -t 01:00:00

# 60 minutes on 15 epochs of 4000 examples for Llama-2-7B on 4 H100 GPUs

export PROJECT_DIR=

# adjust for different experiments
model_name=llama2_7b_base                                         # llama2_13b_base, mistral_7b_v0.2_base, llama3_8b_base
k=2                                                               # 1, 3
NUM_TRAIN_EXAMPLES=1000                                           # 2000, 4000
MODEL_STR=${model_name}_sda_k${k}_${NUM_TRAIN_EXAMPLES}           # ${model_name}_sdd_k${k}_${NUM_TRAIN_EXAMPLES}
DATA_PATH=${PROJECT_DIR}/datasets/ism_sda_k${k}_${NUM_TRAIN_EXAMPLES}.json 
n_epochs=15

# adjustable hyperparameters
LR=0.00002
TRUE_BS=64

# load parameters, environment variables
source ./src/slurm/template.sh

# Gemma2 models require MAmmoTH package
if [[ $model_name == *"gemma2"* ]]; then
    cd MAmmoTH
    export MASTER_ADDR="localhost"
    export GLOO_SOCKET_IFNAME="lo"
    export NCCL_SOCKET_IFNAME="lo"
    export WANDB_MODE="offline"
    export WANDB_DISABLED="true"

    torchrun \
        --master_addr ${MASTER_ADDR} \
        --nproc_per_node ${N_GPUS} \
        --master_port ${MASTER_PORT} \
        train.py \
            --model_name_or_path ../base_models/gemma-2-2b \
            --data_path ${DATA_PATH} \
            --template_variation False \
            --bf16 True \
            --output_dir ${CHECKPOINT_PATH} \
            --num_train_epochs ${N_EPOCHS} \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps ${ACCUM_STEPS} \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps $(($NUM_TRAIN_EXAMPLES / $TRUE_BS)) \
            --learning_rate ${LR} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --fsdp "full_shard auto_wrap" \
            --fsdp_transformer_layer_cls_to_wrap 'Gemma2DecoderLayer' \
            --flash_attn False
else

# all other models can be trained with the torchtune package
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        mkdir -p "$CHECKPOINT_PATH"
    fi
    tune run \
        --master-port ${MASTER_PORT} \
        --nnodes 1 \
        --nproc_per_node ${N_GPUS} \
        src/train.py \
        --config ${TRAIN_CONFIG_PATH} \
            checkpointer.output_dir=${CHECKPOINT_PATH} \
            dataset.data_files=${DATA_PATH} \
            optimizer.lr=${LR} \
            batch_size=${BATCH_SIZE} \
            gradient_accumulation_steps=${ACCUM_STEPS} \
            epochs=${N_EPOCHS}
fi