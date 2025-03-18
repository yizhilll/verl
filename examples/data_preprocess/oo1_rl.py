# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_instruction_qwen(text: str) -> str:
    try:
        prefix = '<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant'
        start_pos = text.index(prefix) + len(prefix)
        try:
            end_pos = text.index(suffix, start_pos)
        except ValueError:
            end_pos = len(text)
        text = text[start_pos:end_pos]
    except:
        print(f'Could not extract instruction from:', text)
    return text

# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split('#### ')[1].replace(',', '')
#     return final_solution

# pip install orjsonl math_verify
import orjsonl
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/oo1_rl')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--online_hard_uids', default=None)
    # parser.add_argument('--selection_mode', default="hard-only", choices=["hard-only"])

    args = parser.parse_args()

    # example filter run: python -m examples.data_preprocess.oo1_rl --local_dir data/oo1_rl_hard_0.01-0.4 --online_hard_uids data/hard_uids_0.01-0.4.pkl

    data_source_list = [
        '/map-vepfs/yizhi/OpenRLHF/data/openo1-big-math-200K_processed_with-split-cot-sys.json',
        # '/map-vepfs/yizhi/OpenRLHF/data/openo1-RL-V4.1_190K-no-mcq_processed_with-split-cot-sys.json',
        '/map-vepfs/yizhi/OpenRLHF/data/openo1-RL-V3.1_193K_remove-mcq_math-curriculum_processed_with-split-cot-sys.json'
        # '/map-vepfs/yizhi/OpenRLHF/data/RL-Mix-V3.cleaned-for-verifier.jsonl.gz',
    ]
    all_data = []
    for fp in data_source_list:
        if not fp.endswith('.json'):
            tmp_data = orjsonl.load(fp)
        else:
            with open(fp, 'r') as f:        
                tmp_data=json.load(f)
        
        # exclude numina data
        if 'RL-V4.1' in fp or 'RL-V3.1' in fp:
            for item in tmp_data:
                if 'numina' in item['source']:
                    continue
                all_data.append(item)
        else:
            all_data.extend(tmp_data)
        
    del tmp_data
    
    for item in all_data:
        # if item['Topic'] == 'Code':
        #     print(item)
        #     print(item.keys())
        #     break
        if 'cases' in item:
            item['cases'] = json.dumps(item['cases'])
        else:
            item['cases'] = json.dumps([])
    
    if args.online_hard_uids is not None:
        import pickle
        with open(args.online_hard_uids, 'rb') as f:
            uids_to_keep = set(pickle.load(f))
        print(f"About to dump {len(all_data)-len(uids_to_keep)} items")
        all_data = [item for item in all_data if item['uid'] in uids_to_keep]

    dataset = datasets.Dataset.from_list(all_data).shuffle(seed=42).train_test_split(test_size=1024)
    # dataset['train']['cases'][0]
    print(dataset)
    # exit()
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(train_dataset[0])
    
    system_cot_split = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <Thought> </Thought> and <Output> </Output> tags, respectively. Note that the actual verifiable answer content is surrounded by \\boxed{}, i.e., <Thought> reasoning process here </Thought><Output> answer here \\boxed{verifiable content here} </Output>.'
    global_data_source = 'oo1_rl'
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('query')
            solution = example.pop('response')
            example['data_type'] = json.dumps(example['data_type'])
            
            question = extract_instruction_qwen(question)
            topic = example.pop('Topic')
            data = {
                "data_source": global_data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_cot_split,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": topic,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_source': example.pop('source'),
                    'Topic': topic,
                    'answer_type': example.pop('answer_type'),
                    'quality': example.pop('quality'),
                    'difficulty': example.pop('difficulty'),
                    'uid': example.pop('uid'),
                    'cases': example.pop('cases'),
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=64)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    # get the unique data_type in the dataset, and
    # pick 8 samples from each 'data_type' for debugging
    t = set(train_dataset['data_type'])
    sample_ds_list = []
    for data_type in t:
        print(f"Data type: {data_type}")
        filter_dataset = train_dataset.filter(lambda x: x['data_type'] == data_type and len(x['prompt'][1]["content"])<300)
        sample_ds_list.append(
            filter_dataset.select(range(min(512, len(filter_dataset))))
        )
        print(f"example: {sample_ds_list[-1][0]}")
    sample_ds = datasets.concatenate_datasets(sample_ds_list)
    sample_ds.to_parquet(os.path.join(local_dir, 'sample.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
