Our code in this directory is adapted from the original repo [WildBench](https://github.com/allenai/WildBench/tree/b39dbd762ba620da5a307cb9cfe8ce56a9fabd0c)

## Changes Made

`src/eval.py` (line 20, lines 87-88, lines 207-208, lines 397-398)
`src/unified_utils.py` (line 11, lines 21-26, line 33, lines 374-440)
`src/openai_batch_eval/instant_results_format.py`

- add support for OpenAI API calls (non-batch)
- add support for AzureOpenAI client
- remove support for unnecessary clients / packages


`src/fastchat_conversation.py` (lines 13-18, line 89, lines 225-239, lines 1512-1526)

- add support for models we finetuned


`src/hf_models.py` (lines 144-145)

- add support for Gemma models which were not compatible with SDPA attention implementation


`src/unified_infer.py` (line 30, lines 139-140)

- add support for Llama3 models which need explicit `stop_token_ids` tag for better generations


`src/eval.py` (lines 175-177, lines 183-185, line 252, line 304)
`src/unified_utils.py` (lines 254-255)

- miscellaneous debugging


`src/view_wb_eval.py` (lines 6-7, lines 29-30, line 42)

- add code to select the directory for saving the model generation and the annotation results
- skip raw files