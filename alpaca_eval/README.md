Our code in this directory is adapted from the original repo [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval/tree/v0.6.2)

## Changes Made

`src/alpaca_eval/main.py` (lines 368-375)

- isloate the generation code from evaluation/annotation code to be compatible with HPC clusters without Internet access on compute nodes