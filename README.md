## Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning

This repository contains the code for our paper [Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning](https://arxiv.org/abs/2408.14774)

**************************** **Updates** ****************************
* 01/22/2025: [Our paper](https://openreview.net/forum?id=44z7HL4mfX) is accepted as a poster in ICLR 2025. See you in Singapore!

## Quick Links

- [Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning](#instruct-skillmix)
- [Quick Links](#quick-links)
- [Overview](#overview)
- [Main Results](#main-results)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
  - [Generate Synthetic Data](#generate-synthetic-data)
  - [Train Models](#train)
  - [Select Model Checkpoint](#choose-model-checkpoint)
  - [Evaluate Models](#evaluate)
- [Model links](#model-links)
- [Dataset links](#dataset-links)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview

## Main Results

## Experiments

In the following section, we provide instructions on reproducing the experiments in our paper.

### Prepare Conda Environment

First, set the following bash variable based on your machine and update the following files.
```Shell
PROJECT_DIR="/absolute path to the project folder/Instruct-SkillMix"
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/1_finetune.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/2_validate.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/3_validate_annotate.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/4_evaluate_AlpacaEval.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/4_evaluate_lm-evaluation-harness.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/4_evaluate_MTBench.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/4_evaluate_WildBench.sh
sed -i "s#export PROJECT_DIR=#export PROJECT_DIR=${PROJECT_DIR}#" $PROJECT_DIR/src/slurm/5_annotate.sh
```

Then prepare a conda environment using the following commands
```Shell
cd $PROJECT_DIR
conda create -n Instruct-SkillMix python=3.11 -y
conda activate Instruct-SkillMix
pip install pip==24.3.1  # enable PEP 660 support 
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install flashinfer==0.1.3 -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install -r requirements.txt
pip install -e lm-evaluation-harness # requirement: pip < 25.0
cd FastChat
pip install -e ".[model_worker,llm_judge]" # requirement: pip < 25.0
CONDA_ENV="/absolute path to the conda environment/Instruct-SkillMix"
cp alpaca_eval/main.py ${CONDA_ENV}/lib/python3.11/site-packages/alpaca_eval/main.py
```

### Prepare API Keys

Insert your own API keys to `src/openai_configs.yaml` (for AlpacaEval annotations) and `src/api_keys.sh` (for data generation and all other annotations).

### Generate Synthetic Data (TBD)

- Generate data
- Convert heldout data to AlpacaEval format (for checkpoint selection)

### Train Models

First, download relevant base models
```Shell
cd $PROJECT_DIR
conda activate Instruct-SkillMix
tune download meta-llama/Llama-2-7b-hf --output-dir ./base_models/Llama-2-7b-hf
tune download meta-llama/Llama-2-13b-hf --output-dir ./base_models/Llama-2-13b-hf
tune download mistral-community/Mistral-7B-v0.2 --output-dir ./base_models/Mistral-7B-v0.2 --ignore-patterns None
tune download meta-llama/Meta-Llama-3-8B --output-dir ./base_models/Meta-Llama-3-8B --ignore-patterns "original/consolidated.00.pth" 
## if you do not have hf_token saved in your bash configuration files, also append --hf-token <HF_TOKEN> above
mv ./base_models/Meta-Llama-3-8B/original/tokenizer.model ./base_models/Meta-Llama-3-8B

tune download google/gemma-2-9b --output-dir ./base_models/gemma-2-9b  --ignore-patterns None
tune download google/gemma-2-2b --output-dir ./base_models/gemma-2-2b  --ignore-patterns None
```

Then, edit the relevant slurm script `src/slurm/1_finetune.sh` and run 
```Shell
cd $PROJECT_DIR
sbatch src/slurm/1_finetune.sh
```

### Select Model Checkpoint
We select the best checkpoint based on the performance on heldout data. Edit the relevant slurm script `src/slurm/2_validate.sh` and run 
```Shell
cd $PROJECT_DIR
sbatch src/slurm/2_validate.sh
```

Then evaluate each checkpoint by editing and running `src/slurm/3_validate_annotate.sh`. Select the best model checkpoint from e.g., `results/ism_sda_k2_heldout/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv`.

### Evaluate Models
Edit the relevant slurm scripts `src/slurm/4_evaluate_*.sh` and generate model outputs by running
```Shell
cd $PROJECT_DIR
sbatch src/slurm/4_evaluate_AlpacaEval.sh
sbatch src/slurm/4_evaluate_MTBench.sh
sbatch src/slurm/4_evaluate_lm-evaluation-harness.sh
sbatch src/slurm/4_evaluate_WildBench.sh
```

Then annotate the model outputs by editing and running `src/slurm/5_annotate.sh`.

You can view the evaluation outputs by checking `./results/alpaca_eval_2/leaderboards/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv` for AlpacaEval and each folder in `./results/lm-evaluation-harness/model_outputs` for lm-evaluation-harness. For MTBench and WildBench, run
```Shell
cd $PROJECT_DIR
python FastChat/fastchat/llm_judge/show_result.py --judge-model gpt-4-0613 --input-file ./results/mt_bench/model_judgment/gpt-4-0613_single.jsonl
python WildBench/src/view_wb_eval.py ./results/wild_bench_v2 pairwise-gpt4t -1
```

## Links

We release links to our trained models and generated dataset for easier experimentations in the future.

|    Name                                                |    Link                                                                         |
|--------------------------------------------------------|---------------------------------------------------------------------------------|
|    Meta-Llama-3-8B trained on Instruct-SkillMix-SDA    |   [Link](https://huggingface.co/PrincetonPLI/Llama-3-8B-Instruct-SkillMix)      |
|    Instruct-SkillMix-SDD Dataset                       |   [Link](https://huggingface.co/datasets/PrincetonPLI/Instruct-SkillMix-SDD)    |
|    Instruct-SkillMix-SDA Dataset                       |   [Link](https://huggingface.co/datasets/PrincetonPLI/Instruct-SkillMix-SDA)    |


## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Simran (skaur 'at' princeton 'dot' edu) and Simon (juhyunp 'at' princeton 'dot' edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give more effective help!

## Citation

Please cite our paper if you find our paper or this repo helpful:
```bibtex
@misc{kaur2024instructskillmixpowerfulpipelinellm,
      title={Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning}, 
      author={Simran Kaur and Simon Park and Anirudh Goyal and Sanjeev Arora},
      year={2024},
      eprint={2408.14774},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.14774}, 
}
```