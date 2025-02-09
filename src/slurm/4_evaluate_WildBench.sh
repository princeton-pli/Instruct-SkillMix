#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint gpu80
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-gpu 12
#SBATCH --output=logs/%x-%j-%a.out
#SBATCH --job-name=evaluate_WildBench
#SBATCH -t 01:00:00

# 30 minutes for Llama-2-7B on 4 H100 GPUs

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
    
## WildBench
cd ${PROJECT_DIR}/WildBench
model_pretty_name=checkpoints_${MODEL_STR_EXTENDED}

# Data-parallellism
start_gpu=0
num_gpus=1
n_shards=${N_GPUS}
shard_size=$((1024 / ${n_shards}))
output_dir=${EVAL_PATH}/wild_bench_v2/model_answer
shards_dir=${output_dir}/tmp_${model_pretty_name}

# Gemma2 models not fully compatible with vllm engine
if [[ $model_name == *"gemma2"* ]]; then
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --start_index $start --end_index $end \
        --data_name wild_bench \
        --model_name ${CHECKPOINT_PATH}/epoch_${epoch} \
        --engine hf \
        --download_dir "default" \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p 1.0 --temperature 0.9 \
        --batch_size 1 --max_tokens 4096 \
        --output_folder $shards_dir &
done 
wait 

# Explicitly setting stop_token_ids to end of token ids for better generations
elif [[ $model_name == *"llama3"* ]]; then
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --start_index $start --end_index $end \
        --data_name wild_bench \
        --model_name ${CHECKPOINT_PATH}/epoch_${epoch} \
        --download_dir "default" \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p 1.0 --temperature 0.9 \
        --batch_size 1 --max_tokens 4096 \
        --stop_token_ids 128001,128009 \
        --output_folder $shards_dir &
done 
wait 

else
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --start_index $start --end_index $end \
        --data_name wild_bench \
        --model_name ${CHECKPOINT_PATH}/epoch_${epoch} \
        --download_dir "default" \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --top_p 1.0 --temperature 0.9 \
        --batch_size 1 --max_tokens 4096 \
        --output_folder $shards_dir &
done 
wait 

fi

python src/merge_results.py $shards_dir epoch_${epoch}
mv $shards_dir/epoch_${epoch}.json $output_dir/${model_pretty_name}.json