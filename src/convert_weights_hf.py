import json
import torch
import os
import shutil
import argparse
import yaml

def convert_number_format(num, num_chars=4):
    num = str(num)
    while len(num) < num_chars:
        num = f"0{num}"
    return num

def main(args):
    print(args.checkpoint_dir,flush=True)

    with open(args.train_config_path, 'r') as file:
        train_config_path = yaml.safe_load(file)
    num_shards =  len(train_config_path['checkpointer']['checkpoint_files'])
    
    shard_ids = [convert_number_format(idx+1) for idx in range(num_shards)]
    shard_files = [f"{args.checkpoint_dir}/hf_model_{shard_id}_{args.epoch-1}.pt" for shard_id in shard_ids]
    
    # create the output dictionary
    output_dict = {"weight_map": {}, "metadata": {}}
    for shard_file in shard_files:
        sd = torch.load(shard_file, mmap=True, map_location='cpu')
        shard = shard_file.split('/')[-1]
        for key in sd.keys():
            output_dict['weight_map'][key] = shard

    new_checkpoint_dir = f"{args.checkpoint_dir}/epoch_{args.epoch}"
    os.makedirs(new_checkpoint_dir, exist_ok=True)
    print(new_checkpoint_dir,flush=True)


    with open(f'{new_checkpoint_dir}/pytorch_model.bin.index.json', 'w') as f:
        json.dump(output_dict, f)

    for shard_file in shard_files:
        shutil.move(shard_file, new_checkpoint_dir)
    config = f"{args.checkpoint_dir}/config.json"
    shutil.copy(config, new_checkpoint_dir)

    
    tokenizer_path = train_config_path['tokenizer']['path'].split('/tokenizer.model')[0]
    tokenizer_files = ["special_tokens_map.json","tokenizer_config.json","tokenizer.json","tokenizer.model"]
    for tokenizer_file in tokenizer_files:
        shutil.copy(f"{tokenizer_path}/{tokenizer_file}", new_checkpoint_dir)
        

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--train_config_path", type=str, default="")

    args = parser.parse_args()

    main(args)