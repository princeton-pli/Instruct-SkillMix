source src/slurm/template.sh
source ${API_KEYS}

P=4
k=2
SEED=42
NUM_ORIG_DATA=1000
NUM_DATA=200

SAVE_BASE_DIR="./datasets"
SAVE_PREFIX="ism_sda"
JUNK_TYPE="junk"

GENERATOR_MODEL="gpt-4-turbo-2024-04-09"
# GENERATOR_MODEL="claude-3-5-sonnet-20240620"

PROMPT_DIR="./src/data_generation/prompts/"
PROMPT_VERSION="generate_${GENERATOR_MODEL}_${JUNK_TYPE}"


## repeat just in case there are errors during generation
## each run will skip already existing data

for tmp in {0..1}
do
    python src/data_generation/2_generate_junk_data.py \
        --num_thread ${P} \
        --k ${k} \
        --seed ${SEED} \
        --num_data ${NUM_DATA} \
        --load_path ${SAVE_BASE_DIR}/${SAVE_PREFIX}_k${k}_${NUM_ORIG_DATA}.json \
        --save_base_dir ${SAVE_BASE_DIR} \
        --save_prefix ${SAVE_PREFIX}_${JUNK_TYPE} \
        --generator_model ${GENERATOR_MODEL} \
        --prompt_dir ${PROMPT_DIR} \
        --prompt_version ${PROMPT_VERSION} \
        --junk_type ${JUNK_TYPE}
done