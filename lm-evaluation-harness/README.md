Our code in this directory is adapted from the original repo [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b70af4f51481db15ec3068d26cde3d76dd0201e7)

## Changes Made

`lm_eval/models/__init__.py` (line 15)
`lm_eval/models/checkpoints.py` (copied from `lm_eval/models/huggingface.py` and changed lines 68-69, lines 1286-1289)

- add support for models we finetuned


`lm_eval/models/checkpoints.py`

- check line 1331