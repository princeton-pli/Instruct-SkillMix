import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from engine import *


################################################
### HELPER METHODS FOR DATA INPUT / OUTPUT
################################################

def concatenate_json(file_list, output_filename):
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
    
    with open(output_filename, "w") as f:
        json.dump(data_list, f, indent=2)

    return


################################################
### HELPER METHODS FOR PARSING CUSTOM DATA FORMAT
################################################

## parse the file that stores the list of query types
def get_query_and_def(elem):
    elem_lst = elem.split('*')
    elem_lst = [elem.strip() for elem in elem_lst if len(elem.strip()) > 0]
    query, query_def = elem_lst[1].strip(), elem_lst[2].strip()
    return {'query_str': query, 'query_type': f'{query}: {query_def}'}

## parse final (instruction, response) pair from GPT output
def extract_relevant_info_gpt(row):
    instruction, response = None, None

    idx = 3
    try:
        if 'Instruction' not in row[f'output_{idx + 1}'] and 'Response' not in row[f'output_{idx + 1}']:
            response = row[f'output_{idx + 1}']
        else:
            response = row[f'output_{idx + 1}'].split('Response:')[1].strip()
    except Exception as e:
        try:
            if 'Instruction' not in row[f'output_{idx}'] and 'Response' not in row[f'output_{idx}']:
                response = row[f'output_{idx}']
            else:
                response = row[f'output_{idx}'].split('Response:')[1].strip()
        except:
            response = None

    try:
        instruction = row[f'output_{idx}'].split('Response')[0]
        instruction = instruction.split('Instruction:')[1].replace('### Refined', '').replace('###','').strip()
    except Exception as e:
        try:
            instruction = row[f'output_{idx + 1}'].split('Response')[0]
            instruction = instruction.split('Instruction:')[1].replace('### Refined', '').replace('###','').strip()
        except:
            instruction = None

    if instruction[0]=="\"" and instruction[-1]=="\"":
        instruction = instruction[1:-1]

    return instruction, response

## parse final (instruction, response) pair from Claude output
def extract_relevant_info_claude(row, idx):
    instruction, response = None, None

    curr = idx
    while instruction is None and curr >= idx - 2:
        output = row[f'output_{curr}']
        instruction = extract_instruction_claude(output)
        curr -= 1

    curr = idx
    while response is None and curr >= idx - 2:
        output = row[f'output_{curr}']
        response = extract_response_claude(output)
        curr -= 1

    return instruction, response

## parse final instruction from Claude output
def extract_instruction_claude(output):
    output_split = output.split("###")
    for output_ in output_split[1:]:
        ## went too far
        if "response:" in output_.lower():
            break 
        ## found it
        if "instruction:" in output_.lower():
            try:
                instruction = output_.split("Instruction:")[1].strip()
            except:
                instruction = output_.split("intruction:")[1].strip()
            return instruction

    for output_ in output_split[1:]:
        ## went too far
        if "response:" in output_.lower():
            break
        ## found it
        if "query:" in output_.lower():
            try:
                instruction = output_.split("Query:")[1].strip()
            except:
                instruction = output_.split("query:")[1].strip()
            return instruction
    return None

## parse final response from Claude output
def extract_response_claude(output):
    output_split = output.split("###")
    for output_ in output_split[1:]:
        ## found it
        if "response:" in output_.lower():
            try:
                response = output_.split("Response:")[1].strip()
            except:
                response = output_.split("response:")[1].strip()
            return response
    return None

################################################
### HELPER METHODS FOR PREPARING DATA
################################################

## get "num" combination of "k" skills from "skill_list"
def get_combinations(skill_list, k, num=-1):
    if num > 0:
        res = []
        for i in range(num):
            np.random.shuffle(skill_list)
            res.append(skill_list[:k])
        return res
    else:
        res=sorted(combinations(skill_list, k))
        res = [list(elem) for elem in res]
        return res

## create a list of subarrays (chunks) to save intermediate results
def chunk_list(original_list, chunk_size):
    chunks = [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]
    return chunks

################################################
### HELPER METHODS FOR GENERATING DATA
################################################

## a wrapper function for generating all data points within each chunk
def generate_chunk(generator_model, save_dir, prompt_version, chunk):
    if "azure-" in generator_model:
        engine = AzureOpenAIChatbotEngine(generator_model.replace("azure-", ""))
    elif "gpt" in generator_model:
        engine = OpenAIChatbotEngine(generator_model)
    elif "claude" in generator_model:
        engine = ClaudeChatbotEngine(generator_model)
    else:
        raise ValueError("unrecognized generator model")

    save_dir = os.path.join(save_dir, "chunk_{}".format(chunk['idx']))
    os.makedirs(save_dir, exist_ok=True)
    
    ## index of the data point within the chunk
    idx = 0
    for (skills, queries) in zip(chunk['comb_list'], chunk['query_list']):
        np.random.shuffle(skills) ## shuffle skills within each combination
        skills_joined_str = ', '.join(skills)
        generate_data(idx, skills_joined_str, queries, engine, save_dir, prompt_version)
        idx += 1
    
    all_save_files = [os.path.join(save_dir, "{}.json".format(idx)) for idx in range(len(chunk['comb_list']))]
    aggregated_save_file = os.path.join(save_dir, "records.json")
    concatenate_json(all_save_files, aggregated_save_file)

## the main function for generating a single data point
def generate_data(idx, skills_joined_str, queries, engine, save_dir, prompt_version):

    temp_file = os.path.join(save_dir, "{}.json".format(idx))
    if os.path.exists(temp_file):
        with open(temp_file) as f:
            data = json.load(f)
        if 'instruction' in data and "output" in data:
            print(f"skipping {idx}, already exists")
            return
            
    num_skills = len(skills_joined_str.split(','))
    query_str = queries['query_str']
    query_type = queries['query_type']

    prompt_file = os.path.join(args.prompt_dir, "{}.json".format(prompt_version))
    with open(prompt_file) as f:
        prompts = json.load(f)
        all_prompts = [p.format(
            num_skills=num_skills,
            skills_str=skills_joined_str,
            query_str=query_str,
            query_type=query_type
        ) for p in prompts]

    if not os.path.exists(temp_file):
        data = {"skills_joined_str": skills_joined_str, 'query_str': query_str}
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(temp_file) as f:
            data = json.load(f)

    conv = engine.initialize_conversation()
    for i, _msg in enumerate(all_prompts):
        conv.append_message(conv.roles[0], _msg)
        conv.append_message(conv.roles[1], None)

        if f"output_{i}" in data:
            conv.update_last_message(data[f"output_{i}"])
            continue

        outputs, prompt = engine.query(conv)
        outputs = outputs.replace("### Query:", "### Instruction:")
        print("INDEX: {}\tTOTAL USAGE: {:.4f}".format(idx, engine.compute_usage()), flush=True)
        conv.update_last_message(outputs)

        data[f'prompt_{i}'] = prompt
        data[f'output_{i}'] = outputs

        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)

    final_idx = len(all_prompts) - 1
    if f"output_{final_idx}" in data:
        try:
            if "gpt" in args.generator_model:
                instruction, response = extract_relevant_info_gpt(data)
            elif "claude" in args.generator_model:
                instruction, response = extract_relevant_info_claude(data, final_idx)

            assert not instruction is None
            assert not response is None
        except:
            data.pop(f"output_{final_idx}", None)
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            return
        
    data["instruction"] = instruction
    data["input"] = ""
    data["output"] = response
    data["generator"] = args.generator_model

    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)

    return


################################################
### MAIN METHOD
################################################

def main(args):
    ## get list of skills / list of queries extracted in the previous step
    extracted_skills = Path(os.path.join(args.save_base_dir, args.generator_model, "skills.txt")).read_text().split('\n') 
    extracted_queries = Path(os.path.join(args.save_base_dir, args.generator_model, "queries.txt")).read_text().split('###') 
    extracted_queries = [get_query_and_def(elem) for elem in extracted_queries if len(elem.strip())>0]

    ## randomly select a skill combination and a query type for each data point
    comb_list = get_combinations(extracted_skills, args.k, args.num_data)
    num_data = len(comb_list) ## could be different from args.num_data if args.num_data == -1
    while len(extracted_queries) < num_data:
        extracted_queries += extracted_queries
    np.random.shuffle(extracted_queries)
    extracted_queries = extracted_queries[:num_data]


    save_dir = os.path.join(args.save_base_dir, args.generator_model, f'k{args.k}_s{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    ## chunk the list of skill combinations / query types 
    chunked_comb_list = chunk_list(comb_list, args.chunk_size)
    chunked_query_list = chunk_list(extracted_queries, args.chunk_size)
    generate_chunk_fn = partial(generate_chunk, args.generator_model, save_dir, args.prompt_version)
    payloads = [{'idx': idx, 'comb_list': comb_list, 'query_list': query_list} for idx, (comb_list, query_list) in enumerate(zip(chunked_comb_list, chunked_query_list))]


    all_data = []
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        all_data = list(executor.map(generate_chunk_fn, payloads))

    all_save_files = [f"{save_dir}/chunk_{idx}/records.json" for idx in range(len(chunked_comb_list))]
    aggregated_save_file = os.path.join(args.save_base_dir, "{}_k{}_{}.json".format(args.save_prefix, args.k, args.num_data))
    concatenate_json(all_save_files, aggregated_save_file)

    print(f'saving to {aggregated_save_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_thread",            type=int, default=4, help="number of parallel threads to call API")
    parser.add_argument("--k",                     type=int, default=1, help="number of skills to combine in each data point")
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--chunk_size",            type=int, default=100, help="size of each intermediate file to save")
    parser.add_argument("--num_data",              type=int, default=-1, help="number of data points to generate, -1 means all possible combinations")
    parser.add_argument("--save_base_dir",         type=str, help='base directory for saving the resulting dataset')
    parser.add_argument("--save_prefix",           type=str, help='prefix to the filename of the resulting dataset', default=None)
    parser.add_argument("--generator_model",       type=str, help="name of generator model, for Azure client, append azure- in the beginning")
    parser.add_argument("--prompt_dir",            type=str, help="directory where prompts are stored")
    parser.add_argument("--prompt_version",        type=str, help="name of the prompt file")

    args = parser.parse_args()

    np.random.seed(args.seed)

    ## if prefix not provided, replace with name of the generator model
    if not args.save_prefix:
        args.save_prefix = args.generator_model
    
    main(args)
