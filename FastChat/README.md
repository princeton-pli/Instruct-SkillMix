Our code in this directory is adapted from the original repo [FastChat](https://github.com/lm-sys/FastChat/tree/d2bc0933c273fb92ecd02281f5f4cbe415c70383)

## Changes Made

`pyproject.toml` (line 25)
`fastchat/llm_judge/common.py` (line 21, lines 169-170, lines 275-277, lines 413-416, lines 421-445, lines 453-460, lines 469-498, line 734)
`fastchat/llm_judge/gen_judgment.py` (line 247)
`fastchat/model/model_adapter.py` (lines 86-102, line 1122, lines 1135-1145, line 2523)

- upgrade to `openai>=1`
- add support for OpenAI, AzureOpenAI clients


`fastchat/llm_judge/gen_judgment.py` (lines 177-178, lines 215-222, line 249, line 257)
`fastchat/llm_judge/gen_model_answer.py` (line 232, lines 281-286)

- add code to select the directory for saving the model generation and the annotation results


`fastchat/llm_judge/gen_judgment.py` (line 315)

- remove code to receive user input to be compatible with HPC clusters 


`fastchat/conversation.py` (line 27, lines 1183-1259)
`fastchat/model/model_adapter.py` (line 64, lines 1178-1179)

- add support for Claude 3.5 Sonnet (copied from [an updated version of FastChat](https://github.com/lm-sys/FastChat/blob/2c68a13bfe10b86f40e3eefc3fcfacb32c00b02a/fastchat/conversation.py))


`fastchat/conversation.py` (line 58, lines 175-189, lines 1636-1650)
`fastchat/model/model_adapter.py` (line 128, line 245, lines 2442-2496)

- add support for models we finetuned


