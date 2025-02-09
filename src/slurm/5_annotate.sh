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

# initalize an empty leaderboard, if it doesn't exist already
if [ ! -d "${EVAL_PATH}/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo" ]; then
    mkdir -p ${EVAL_PATH}/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo
fi
if [ ! -d "${EVAL_PATH}/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv" ]; then
    cp ${PROJECT_DIR}/src/leaderboard.csv ${EVAL_PATH}/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
fi

MODEL_STR_EXTENDED=${MODEL_STR}_lr_${LR}_bs_${TRUE_BS}_epoch_${epoch}_of_${N_EPOCHS}

## AlpacaEval        
alpaca_eval evaluate \
    --output_path ${EVAL_PATH}/alpaca_eval_2/leaderboards/annotations_cached/${MODEL_STR_EXTENDED} \
    --model_outputs ${EVAL_PATH}/alpaca_eval_2/model_outputs/${MODEL_STR_EXTENDED}/model_outputs.json \
    --reference_outputs ${EVAL_PATH}/alpaca_eval_2/model_outputs/${MODEL_STR_EXTENDED}/reference_outputs.json \
    --name ${MODEL_STR_EXTENDED} \
    --precomputed_leaderboard ${EVAL_PATH}/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv

## MT Bench
cd ${PROJECT_DIR}/FastChat/fastchat/llm_judge

export GPT_MODEL=gpt-4-0613
source ${API_KEYS}
python gen_judgment.py \
    --model-list checkpoints_${MODEL_STR_EXTENDED} \
    --parallel 2 \
    --judge-model ${GPT_MODEL} \
    --answer-dir ${EVAL_PATH}/mt_bench/model_answer \
    --output-dir ${EVAL_PATH}/mt_bench/model_judgment
python show_result.py \
    --judge-model ${GPT_MODEL} \
    --input-file ${EVAL_PATH}/mt_bench/model_judgment/${GPT_MODEL}_single.jsonl

## WildBench
cd ${PROJECT_DIR}/WildBench
model_pretty_name=checkpoints_${MODEL_STR_EXTENDED}

export GPT_MODEL=gpt-4-turbo-2024-04-09
source ${API_KEYS}
for ref_name in gpt-4-turbo-2024-04-09
do
    eval_template=evaluation/eval_template.pairwise.v2.md
    eval_folder=${EVAL_PATH}/wild_bench_v2/pairwise.v2/eval=${GPT_MODEL}/ref=${ref_name}
    echo "Evaluating ${MODEL_STR_EXTENDED} vs $ref_name using $GPT_MODEL with $eval_template"
    mkdir -p $eval_folder

    # Data-parallellism
    num_shards=4
    echo "Using $num_shards shards"
    shard_size=$((1024 / $num_shards))
    echo "Shard size: $shard_size"
    start_gpu=0 # not used 
    for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $num_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
        eval_file="${eval_folder}/${model_pretty_name}.$start-$end.json"
        echo "Evaluating ${model_pretty_name} vs $ref_name from $start to $end"
        python src/eval.py \
            --action eval \
            --model ${GPT_MODEL} \
            --max_words_to_eval 1000 \
            --mode pairwise \
            --eval_template $eval_template \
            --target_model_name ${model_pretty_name} \
            --ref_model_name $ref_name \
            --local_result_file ${EVAL_PATH}/wild_bench_v2/model_answer/${model_pretty_name}.json \
            --eval_output_file $eval_file \
            --start_idx $start --end_idx $end &
    done 
    # Wait for all background processes to finish
    wait

    # Run the merge results script after all evaluation scripts have completed
    python src/merge_results.py $eval_folder ${model_pretty_name}
    python src/openai_batch_eval/instant_results_format.py ${eval_folder}/${model_pretty_name}.json
done