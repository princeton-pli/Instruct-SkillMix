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

# initalize an empty leaderboard, if it doesn't exist already
if [ ! -d "${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/weighted_alpaca_eval_gpt4_turbo" ]; then
    mkdir -p ${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/weighted_alpaca_eval_gpt4_turbo
fi
if [ ! -d "${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv" ]; then
    cp ${PROJECT_DIR}/src/leaderboard.csv ${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv
fi


for epoch in {1..15}
do

MODEL_STR_EXTENDED=${MODEL_STR}_lr_${LR}_bs_${TRUE_BS}_epoch_${epoch}_of_${N_EPOCHS}

## AlpacaEval        
alpaca_eval evaluate \
    --output_path ${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/annotations_cached/${MODEL_STR_EXTENDED} \
    --model_outputs ${EVAL_PATH}/${HELDOUT_DATA_NAME}/model_outputs/${MODEL_STR_EXTENDED}/model_outputs.json \
    --reference_outputs ${EVAL_PATH}/${HELDOUT_DATA_NAME}/model_outputs/${MODEL_STR_EXTENDED}/reference_outputs.json \
    --name ${MODEL_STR_EXTENDED} \
    --precomputed_leaderboard ${EVAL_PATH}/${HELDOUT_DATA_NAME}/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv

done