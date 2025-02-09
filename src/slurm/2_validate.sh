#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --job-name=validate
#SBATCH -t 00:20:00
#SBATCH --array 1-15

# 10 minutes per checkpoint for Llama-2-7B on 1 H100 GPUs

# adjust for different experiments
model_name=llama2_7b_base                                         # llama2_13b_base, mistral_7b_v0.2_base, llama3_8b_base
k=2                                                               # 1, 3
NUM_TRAIN_EXAMPLES=1000                                           # 2000, 4000
MODEL_STR=${model_name}_sda_k${k}_${NUM_TRAIN_EXAMPLES}           # ${model_name}_sdd_k${k}_${NUM_TRAIN_EXAMPLES}
DATA_PATH=${PROJECT_DIR}/datasets/ism_sda_k${k}_${NUM_TRAIN_EXAMPLES}.json
HELDOUT_DATA_NAME=ism_sda_k${k}_heldout                           # ism_sdd_k${k}_heldout
n_epochs=15

# adjustable hyperparameters
LR=0.00002
TRUE_BS=64

# load parameters, environment variables
source ./src/slurm/template.sh

epoch=$SLURM_ARRAY_TASK_ID

# Gemma2 models already stored in HF format
if [[ $model_name == *"gemma2"* ]]; then
    NUM_STEPS=$((($NUM_TRAIN_EXAMPLES / $TRUE_BS) * ${epoch}))
    mv ${CHECKPOINT_PATH}/checkpoint-${NUM_STEPS} ${CHECKPOINT_PATH}/epoch_${epoch}
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/optimizer.bin
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/pytorch_model_fsdp.bin
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/rng_state_0.pth
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/rng_state_1.pth
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/rng_state_2.pth
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/rng_state_3.pth
    rm ${CHECKPOINT_PATH}/epoch_${epoch}/scheduler.pt
else 

# torchtune models need to be converted to HF format
    python src/convert_weights_hf.py --checkpoint_dir ${CHECKPOINT_PATH} --epoch ${epoch} --train_config_path ${TRAIN_CONFIG_PATH}
fi

# create a new evaluation config file for the new checkpoint
MODEL_STR_EXTENDED=${MODEL_STR}_lr_${LR}_bs_${TRUE_BS}_epoch_${epoch}_of_${N_EPOCHS}
echo ${MODEL_STR_EXTENDED}
EVAL_CONFIG_DIR=${EVAL_CONFIG_PATH}/sft_models/${MODEL_STR_EXTENDED}
mkdir -p ${EVAL_CONFIG_DIR}
cp ${EVAL_CONFIG_PATH}/hf_checkpoints/${model_name}.yaml ${EVAL_CONFIG_DIR}/configs.yaml

sed -i "s#${model_name}:#${MODEL_STR_EXTENDED}:#" ${EVAL_CONFIG_DIR}/configs.yaml
sed -i "s#prompt_template:#prompt_template: ${PROJECT_DIR}/src/eval_configs/templates/prompt.txt#" ${EVAL_CONFIG_DIR}/configs.yaml
sed -i "s#model_name: '${model_name}'#model_name: '${CHECKPOINT_PATH}/epoch_${epoch}'#" ${EVAL_CONFIG_DIR}/configs.yaml
sed -i "s#pretty_name: '${model_name}'#pretty_name: '${MODEL_STR_EXTENDED}'#" ${EVAL_CONFIG_DIR}/configs.yaml

## run AlpacaEval with heldout data
alpaca_eval evaluate_from_model \
    --model_configs ${EVAL_CONFIG_DIR} \
    --output_path ${EVAL_PATH}/${HELDOUT_DATA_NAME}/model_outputs/${MODEL_STR_EXTENDED} \
    --evaluation_dataset ${PROJECT_DIR}/datasets/${HELDOUT_DATA_NAME}.json