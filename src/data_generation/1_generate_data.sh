P=4
k=2
SEED=42
CHUNK_SIZE=100
NUM_DATA=1000

SAVE_BASE_DIR="./datasets"
SAVE_PREFIX="ism_sda"

GENERATOR_MODEL="gpt-4-turbo-2024-04-09"
# GENERATOR_MODEL="claude-3-5-sonnet-20240620"

PROMPT_DIR="./src/data_generation/prompts/"
PROMPT_VERSION="generate_${GENERATOR_MODEL}"


## repeat just in case there are errors during generation
## each run will skip already existing data

for tmp in {0..3}
do
    python src/data_generation/1_generate_data.py \
        --num_thread ${P} \
        --k $k \
        --seed ${SEED} \
        --chunk_size ${CHUNK_SIZE} \
        --num_data ${NUM_DATA} \
        --save_base_dir ${SAVE_BASE_DIR} \
        --save_prefix ${SAVE_PREFIX} \
        --generator_model ${GENERATOR_MODEL} \
        --prompt_dir ${PROMPT_DIR} \
        --prompt_version ${PROMPT_VERSION}
done