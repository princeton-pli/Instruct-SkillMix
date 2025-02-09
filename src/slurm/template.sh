# load environment
module purge
module load anaconda3/2023.9
export PYTHONUSERBASE=/dev/null
conda activate Instruct-SkillMix

## computing optimization parameters
BATCH_SIZE=4
N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"
ACCUM_STEPS=$((TRUE_BS / BATCH_SIZE / N_GPUS))
N_STEPS_PER_EPOCH=$((NUM_TRAIN_EXAMPLES / TRUE_BS))
N_EPOCHS=${n_epochs:-15}
MAX_STEPS=$((N_EPOCHS * N_STEPS_PER_EPOCH))

## set training arguments
JOB_ID_SUFFIX=${SLURM_JOB_ID: -4}
MASTER_PORT=$((29000 + JOB_ID_SUFFIX + SLURM_ARRAY_TASK_ID))

# set various paths
export SOURCE_PATH=${PROJECT_DIR}/src
export TRAIN_CONFIG_PATH=${PROJECT_DIR}/src/train_configs/${model_name}.yaml
export EVAL_CONFIG_PATH=${PROJECT_DIR}/src/eval_configs
export OPENAI_CLIENT_CONFIG_PATH=${PROJECT_DIR}/src/openai_configs.yaml ## API keys in AlpacaEval format
export API_KEYS=${PROJECT_DIR}/src/api_keys.sh                          ## API keys for all other packages

## path to result files (e.g., checkpoints / evaluation results); recommend scratch drive
export RESULTS_PATH=${PROJECT_DIR}
export EVAL_PATH=${RESULTS_PATH}/results
export CHECKPOINT_PATH=${RESULTS_PATH}/checkpoints/${MODEL_STR}/lr_${LR}/bs_${TRUE_BS}