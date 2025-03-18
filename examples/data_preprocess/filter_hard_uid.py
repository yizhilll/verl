import orjsonl
from collections import defaultdict
from tqdm import tqdm
import pickle

def filter_hard(reward_list, max_acc=0.4, min_acc=None, min_correct=1):
    
    # too less sample
    if len(reward_list) < 4:
        return False, 'too less sample'
    # all format error
    if sum(reward_list) - (-1.0*len(reward_list)) < 1e-5:
        return False, 'all format error'
    
    n_correct = len([1 for r in reward_list if r >0])
    # never correct
    if n_correct < 1:
        return False, 'never correct'
    # too easy
    if n_correct/len(reward_list) > max_acc:
        return False, 'too easy'
    if n_correct/len(reward_list) < min_acc:
        return False, 'too hard'

    return True, 'pass'

run_name_list = [
    'open-o1-Qwen-7B-base_reinfroce-v0.4.5-pattern-mode-split-cot_KL-0_big-math-250K_32gpu_Bo8-Ep20-len-4k',
    'open-o1-Qwen-7B-base_reinfroce-v0.4.5.1-pattern-mode-cot-split_KL-1e5_data-RL-V3.1-curriculum-verifier-only-193K_48gpu_Bo8-Ep1-len-4k', 
]

all_data = []
for run_name in tqdm(run_name_list):
    all_data.extend(orjsonl.load(f"/map-vepfs/yizhi/OpenRLHF/training_records/{run_name}/merged_records.jsonl"))


uid_to_rewards = defaultdict(list)
for item in tqdm(all_data, desc='aggregating rewards'):
    uid_to_rewards[item['uid']].append(item['rewards'])
print(f"total uids: {len(uid_to_rewards)}")

keep_uids = []
reason_dict = defaultdict(int)
for uid, rewards in tqdm(uid_to_rewards.items(), 'filtering'):
    is_keep, reason = filter_hard(rewards, max_acc=0.4, min_acc=0.01)
    if is_keep:
        keep_uids.append(uid)
    reason_dict[reason] += 1

print(f"keep uids: {len(keep_uids)}")
print(f'filter reasons:\n{reason_dict}')
with open('data/hard_uids_0.01-0.4.pkl', 'wb') as f:
    pickle.dump(keep_uids, f)