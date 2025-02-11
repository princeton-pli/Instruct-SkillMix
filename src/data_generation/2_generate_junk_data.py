import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from engine import *

################################################
### HELPER METHODS FOR DATA INPUT / OUTPUT
################################################

def concatenate_json(file_list, orig_data, output_filename):
    data_list = []

    for filename in file_list:
        assert os.path.exists(filename), filename
        with open(filename) as f:
            data = json.load(f)

        if type(data) is dict:
            data = [data]

        for d in data:
            assert "instruction" in d
            assert "input" in d
            assert "output" in d
            assert "skills_joined_str" in d
            assert "query_str" in d
            assert "generator" in d

            new_data = {
                "instruction": d["instruction"],
                "input": "",
                "output": d["output"],
                "skills_joined_str": d["skills_joined_str"],
                "query_str": d["query_str"],
                "generator": d["generator"]
            }

            data_list.append(new_data)
    
    data_list += orig_data

    with open(output_filename, "w") as f:
        json.dump(data_list, f, indent=2)

    return


################################################
### HELPER METHODS FOR GENERATING DATA
################################################

## a helper method for generating each data
def generate_data(generator_model, save_dir, prompt_version, junk_type, payload):
    if "azure-" in generator_model:
        engine = AzureOpenAIChatbotEngine(generator_model.replace("azure-", ""))
    elif "gpt" in generator_model:
        engine = OpenAIChatbotEngine(generator_model)
    elif "claude" in generator_model:
        engine = ClaudeChatbotEngine(generator_model)
    else:
        raise ValueError("unrecognized generator model")

    idx, data = payload
    temp_file = os.path.join(save_dir, "{}.json".format(idx))

    if os.path.exists(temp_file):
        print(f"skipping {idx}, already exists")
        return

    prompt_file = os.path.join(args.prompt_dir, "{}.json".format(prompt_version))
    with open(prompt_file) as f:
        prompts = json.load(f)
        prompt = prompts[0].format(instruction=data['instruction'])
   
    conv = engine.initialize_conversation()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
        
    outputs, prompt = engine.query(conv)
    print("INDEX: {}\tTOTAL USAGE: {:.4f}".format(idx, engine.compute_usage()), flush=True)
    if outputs[0]=="\"" and outputs[-1]=="\"":
        outputs = outputs[1:-1]

    data["output"] = outputs

    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)

    return
    

################################################
### MAIN METHOD
################################################

def main(args):
    
    with open(args.load_path) as f:
        data = json.load(f)
    orig_num_data = len(data)
    orig_data = data[args.num_data:]
    data = data[:args.num_data]

    save_dir = os.path.join(args.save_base_dir, args.generator_model, f'k{args.k}_s{args.seed}_{args.junk_type}')
    os.makedirs(save_dir, exist_ok=True)

    generate_data_fn = partial(generate_data, args.generator_model, save_dir, args.prompt_version, args.junk_type)
    payloads = [(idx, d) for (idx, d) in enumerate(data)]

    all_data = []
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        all_data = list(executor.map(generate_data_fn, payloads))

    all_save_files = [os.path.join(save_dir, "{}.json".format(idx)) for idx in range(args.num_data)]
    aggregated_save_file = os.path.join(args.save_base_dir, "{}_k{}_{}.json".format(args.save_prefix, args.k, orig_num_data))
    concatenate_json(all_save_files, orig_data, aggregated_save_file)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_thread",            type=int, default=4, help="number of parallel threads to call API")
    parser.add_argument("--k",                     type=int, default=1, help="number of skills to combine in each data point")
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--num_data",              type=int, default=-1, help="number of data points to generate, -1 means all data")
    parser.add_argument("--load_path",             type=str, help='path to load dataset from')
    parser.add_argument("--save_base_dir",         type=str, help='base directory for saving the resulting dataset')
    parser.add_argument("--save_prefix",           type=str, help='prefix to the filename of the resulting dataset', default=None)
    parser.add_argument("--generator_model",       type=str, help="name of generator model, for Azure client, append azure- in the beginning")
    parser.add_argument("--prompt_dir",            type=str, help="directory where prompts are stored")
    parser.add_argument("--prompt_version",        type=str, help="name of the prompt file")
    parser.add_argument("--junk_type",             type=str, help="type of junk data to introduce")

    args = parser.parse_args()

    np.random.seed(args.seed)

    ## if prefix not provided, replace with name of the generator model
    if not args.save_prefix:
        args.save_prefix = args.generator_model
    
    main(args)
