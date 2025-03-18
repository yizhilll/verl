import json
import ray
import numpy as np
from copy import deepcopy
import os
from typing import TypeVar, Dict, List, Union, Any, Tuple, Literal
from pydantic_settings import BaseSettings
import re
import signal


from evaluation.parser import extract_answer, find_box
# from evaluation.grader import math_equal_process, math_equal
# from evaluation.verifier_math_conversion import extract_label_content, remove_outer_parentheses, is_valid_math
from evaluation.code import extract_code_and_test
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
# from openrlhf.utils.data_record_util import JsonlFileHandler


MAX_INS_LEN_AS_KEY = 2400

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timeout")

class Verifier_Settings(BaseSettings):
    """Verifier_Settings class that handles configuration with environment variable override support."""
    pt_model_name: str = "OpenO1/OpenO1-Qwen-7B-v0.1"
    datapath_list: str = "/map-vepfs/openo1/LLaMA-Factory/data/numina.jsonl"
    qa_mapping_file: str = "evaluation/data/templated_numina_qa_mapping.pkl"
    num_verifier_processes: int = 1
    prompt_max_len: int = 1024
    input_key: str = "query"
    output_key: str = "response"
    answer_key: str = 'None'
    maximum_len_reward: float = -1.0
    reward_ref_format: str = 'output_only'
    apply_chat_template: bool = True
    return_additional_info: bool = True
    save_query_response: bool = True
    save_reward_records_folder: str = 'None'
    save_record_batch_size: int = 256
    verifier_base_url: str = 'http://127.0.0.1:15078/get_response_from_llm'
    local_verifier_path: str = 'None'
    tp: int = 8
    voting_k: int = 1
    use_vllm_group: bool = False
    enable_math_expr_extract: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._override_from_env()
        
    @classmethod
    def _get_env_variables(cls) -> Dict[str, Any]:
        env_settings = {}
        settings_fields = cls.__annotations__
        
        for env_var in os.environ:
            if env_var.isupper():
                setting_name = env_var.lower().removeprefix('rm_')
                if setting_name in settings_fields:
                    expected_type = settings_fields[setting_name]
                    value = os.environ[env_var]
                    try:
                        if expected_type == bool:
                            value = value.lower() in ('true', '1', 'yes')
                        else:
                            value = expected_type(value)
                        env_settings[setting_name] = value
                    except ValueError as e:
                        print(f"Warning: Could not convert environment variable {env_var} to type {expected_type}: {e}")
        return env_settings

    def _override_from_env(self) -> None:
        env_settings = self._get_env_variables()
        if env_settings:
            print("Overriding settings with environment variables:")
            for key, value in env_settings.items():
                if hasattr(self, key):
                    print(f"  {key}: {getattr(self, key)} -> {value}")
                    setattr(self, key, value)

# Utility functions
def is_number(element: str) -> bool:
    try:
        float(element.replace(" ", ""))
        return True
    except ValueError:
        return False

def split_output(text: str) -> Tuple[str, str]:
    cot = ""
    output = text.strip()
    
    # Try to find the final answer after "Therefore" or similar keywords
    keywords = ['<Output>']
    for keyword in keywords:
        if keyword.lower() in output.lower():
            parts = output.lower().split(keyword.lower())
            if len(parts) > 1:
                cot = output[:output.lower().rindex(keyword.lower())]
                output = output[output.lower().rindex(keyword.lower()):].strip()
                break
    
    return cot, output

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

def extract_instruction_llama(text: str) -> str:
    text = text.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')[0].strip()
    text = text.split('<|start_header_id|>user<|end_header_id|>')[1].strip()
    return text

def extract_response_qwen(text: str) -> str:
    prefix = '<|im_start|>assistant'
    suffix = '<|im_end|>'
    try:
        start_pos = text.index(prefix) + len(prefix)
    except:
        try:
            start_pos = text.index('<Thought>')
        except:
            start_pos = 0 
    try:
        end_pos = text.index(suffix, start_pos)
    except ValueError:
        end_pos = len(text)
    return text[start_pos:end_pos]

def extract_response_llama(text: str) -> str:
    return text.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()

def format_instruction(text: str) -> str:
    return text[:MAX_INS_LEN_AS_KEY]


UNSUPPORTED_DATA_TYPES = [
    'no_type',
    # ('Math', 'b'),
    ('Math', 'c'),
    ('Reasoning', 'b'),
    # ('mcq_26m_close_form', 'b'),
]

@ray.remote(max_retries=1)
def rule_based_verifier_ray(
    output: str, 
    gold_standard: str,
    enable_math_expr_extract: bool
) -> float:
    return rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)

def rule_based_verifier(
    output: str, 
    gold_standard: str,
    enable_math_expr_extract: bool
) -> float:
    cot, output = split_output(output) # not extrac the boxed at this step
    
    # all in math_verify!
    
    # this could first check on the string match
    answer_string = find_box(output) # find the actual response and remove the boxed{}
    if gold_standard == answer_string:
        return 1.0
    if gold_standard.startswith('$') and gold_standard.endswith('$'):
        if gold_standard[1:-1] == answer_string:
            return 1.0
    # if answer_string.startswith('{') and answer_string.endswith('}') \
    #     and answer_string[1:-1] == gold_standard:
    #         return 1.0
    
    answer_parsed = parse(
                output , # the parse will need to extract the boxed
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            # equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
    answer_string_parsed = parse(
                "\\boxed{" + answer_string + "}",  # the parse will need to extract the boxed
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            # equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
    # preprend parsing sourounding for gold_standards
    if not(gold_standard.startswith('$') and gold_standard.endswith('$')) and not(gold_standard.startswith('\\boxed{') and gold_standard.endswith('}')):
        gold_standard = "\\boxed{" + gold_standard + "}"
    gold_parsed = parse(gold_standard, extraction_mode="first_match")
    if len(gold_parsed) != 0:
        return float(verify(answer_parsed, gold_parsed) or verify(answer_string_parsed, gold_parsed))
    else:
        gold_parsed = parse(f'${gold_standard}$', extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            return float(verify(answer_parsed, gold_parsed) or verify(answer_string_parsed, gold_parsed))
        else:
            assert ValueError, f"Invalid gold standard: {gold_standard}, which should be filtered before training"
    
    

    return 0.0


# 修改 safe_rule_based_verifier 函数
def safe_rule_based_verifier(
    output: str, 
    gold_standard: str, 
    enable_math_expr_extract: bool,
    retry: int = 2,
    timeout_per_try: int = 3,
) -> float:
    current_try = 0
    while current_try < retry:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_try) # `timeout_per_try` must be integer
            reward = rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)
            signal.alarm(0)
            return reward
        except Exception as e:
            print(f'Trying rule_based_verifier() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
            print(f'error output:', e)
            current_try += 1
    
    print(f"[WARNING]: Rule-based verifier failed after {retry} attempts with timeout")
    return 0.0

   
  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_math_rewards(
    ref_responses: List[str],
    pred_responses: List[str],
    enable_math_expr_extract: bool,
    use_ray: bool = True
) -> List[float]:

    if use_ray:
        refs = []
        for pred, ref in zip(pred_responses, ref_responses):
            refs.append(rule_based_verifier_ray.remote(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract))
        rewards = []
        for i, ref in enumerate(refs):
            try:
                rewards.append(ray.get(ref, timeout=1200))
                # rewards.append(ray.get(ref))
            except (ray.exceptions.RayActorError,
                   ray.exceptions.GetTimeoutError,
                   ray.exceptions.RaySystemError,
                   ray.exceptions.ObjectLostError,
                   ray.exceptions.TaskCancelledError,
                   ray.exceptions.RayTaskError,
                   Exception) as e:
                print(f"can't verify {pred_responses[i]}, set reward 0, failed with error: {str(e)}")
                rewards.append(0.0)

    else:
        rewards = []
        for pred, ref in zip(pred_responses, ref_responses):
            reward = safe_rule_based_verifier(output=pred, gold_standard=ref, enable_math_expr_extract=enable_math_expr_extract)
            
            rewards.append(reward)
    return rewards

def safe_code_verifier(idx, entry, extracted_pred, retry=2, timeout_per_try=4) -> float:
    current_try = 0
    while current_try < retry:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_try) # `timeout_per_try` must be integer
            idx, reward = extract_code_and_test((idx, entry, extracted_pred))
            signal.alarm(0)
            return idx, reward
        except:
            print(f'Trying extract_code_and_test() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
            current_try += 1
    
    print(f"[WARNING]: Rule-based extract_code_and_test verifier failed after {retry} attempts with timeout")
    return idx, 0.0

def code_verifier(idx, entry, extracted_pred) -> float:
    idx, reward = extract_code_and_test((idx, entry, extracted_pred))
    return idx, reward

@ray.remote(max_retries=1)
def code_verifier_ray(idx, entry, extracted_pred) -> float:
    return code_verifier(idx, entry, extracted_pred)


def get_code_rewards(entries: List[Dict], pred_responses: List[str], use_ray: bool = True) -> List[float]:
    rewards = []
    if use_ray:
        refs = []
        for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
            cot, extracted_pred = split_output(pred)
            extracted_pred = find_box(extracted_pred)
            refs.append(code_verifier_ray.remote(idx, entry, extracted_pred))
        rewards = []
        for i, ref in enumerate(refs):
            try:
                idx, reward = ray.get(ref, timeout=1200)
                # idx, reward = ray.get(ref)
                rewards.append(reward)
            except (ray.exceptions.RayActorError,
                   ray.exceptions.GetTimeoutError,
                   ray.exceptions.RaySystemError,
                   ray.exceptions.ObjectLostError,
                   ray.exceptions.TaskCancelledError,
                   ray.exceptions.RayTaskError,
                   Exception) as e:
                print(f"can't verify: ...{pred_responses[i][-128:]}. Set reward 0, failed with error: {str(e)}")
                rewards.append(0.0)
    else:
        for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
            cot, extracted_pred = split_output(pred)
            extracted_pred = find_box(extracted_pred)
            idx, reward = safe_code_verifier(idx, entry, extracted_pred)
            rewards.append(reward)
    return rewards

# Core reward calculation functions
def process_mcq_reward(answer_type: str, pred: str, option: str, content: str) -> float:

    patterns = {
        'a': r'(.*?)', # dummy placeholder
        'b': r'<option>\[(.*?)\]</option>',
        'c': r'<<\((.*?)\)>>',
        'd': r'==(.*?)==',
        'e': r'##(.*?)##'
    }
    
    possible_gt = [option, option+'.', option.lower(), option.lower()+'.']
    # if answer_type != 'c':
    possible_gt.extend([content, f'{option}. {content}', f'{option}.{content}'])
    
    if answer_type == 'a':
        pred_content = pred.strip().rstrip()
    else:
        pattern = patterns.get(answer_type)
        if not pattern:
            return 0.0
        match = re.search(pattern, pred, re.IGNORECASE)
        pred_content = match.group(1) if match else ""
        pred_content = pred_content.strip().rstrip()
    return float(pred_content in possible_gt)
    
def get_matching_rewards(entries: List[Dict], pred_responses: List[str]) -> List[float]:
    rewards = []
    
    for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
        ground_truth = entry['response']
        cot, output = split_output(pred)
        # output = extract_answer(pred_str=output, data_name='N/A', use_last_number=False)
        output = find_box(output)
        
        ground_truth = ground_truth.strip().rstrip()
        output = output.strip().rstrip()
        is_label, label, labelled_content = extract_label_content(output)
        if is_label:
            labelled_content.strip().rstrip()
        if (labelled_content == ground_truth) or (output == ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

def get_mcq_rewards(entries: List[Dict], pred_responses: List[str]) -> List[float]:
    rewards = []
    
    for idx, (entry, pred) in enumerate(zip(entries, pred_responses)):
        ground_truth = entry['response']
        cot, output = split_output(pred)
        # output = extract_answer(pred_str=output, data_name='N/A', use_last_number=False)
        output = find_box(output)

        option = None
        ground_truth_content = None
        for i in range(5):
            if ground_truth.startswith(chr(ord('A')+i)):
                option = chr(ord('A')+i)
                ground_truth_content = ground_truth.removeprefix(option+'.').strip().rstrip()
                break
        reward = process_mcq_reward(
            entry['answer_type'], 
            output, 
            option, 
            ground_truth_content
        )
        rewards.append(reward)

    return rewards

def process_entries(
    entries: List[Dict],
    enable_math_expr_extract: bool,
    use_ray: bool = True,
    output_key: str = 'response',
    model_output_key: str = 'model_response',
) -> Dict:

    type_to_ids, _ = get_instruction_type_mapping(entries=entries)
    assert 'no_type'not in type_to_ids
    
    # pred_responses = [entry[model_output_key] for entry in entries]
    rewards = np.zeros(len(entries))
    
    # Handle math equations
    math_indices = type_to_ids[('Math', 'a')] + \
        type_to_ids[('Math', 'b')] + \
                   type_to_ids[('Reasoning', 'a')] + \
                   type_to_ids[('mcq_26m_close_form', 'a')] + \
                   type_to_ids[('mcq_26m_close_form', 'b')]
                
    if math_indices:
        math_rewards = get_math_rewards(
            ref_responses=[entries[i][output_key] for i in math_indices],
            pred_responses=[entries[i][model_output_key] for i in math_indices],
            enable_math_expr_extract=enable_math_expr_extract,
            use_ray=use_ray,
        )
        rewards[math_indices] = math_rewards

    # Handle code evaluation
    code_indices = type_to_ids[('Code', 'assert')] + type_to_ids[('Code', 'input')]
    if code_indices:
        code_rewards = get_code_rewards(
            entries=[entries[i] for i in code_indices],
            pred_responses=[entries[i][model_output_key] for i in code_indices],
            use_ray=use_ray,
        )
        rewards[code_indices] = code_rewards

    # Handling MCQ rewards
    mcq_indices = []
    for i in range(5):
        mcq_indices.extend(type_to_ids[('mcq_26m', chr(ord('a')+i))])
    if mcq_indices:
        mcq_rewards = get_mcq_rewards(
            entries=[entries[i] for i in mcq_indices],
            pred_responses=[entries[i][model_output_key] for i in mcq_indices],
        )
        rewards[mcq_indices] = mcq_rewards

    # Handle matching cases
    matching_indices = []
    for entry_id in type_to_ids[('Reasoning', 'a')]:
        if rewards[entry_id] == 0.0: # not matched by math-verify
            matching_indices.append(entry_id)
    # matching_indices.extend(type_to_ids[('Reasoning', 'a')])
    if matching_indices:
        matching_rewards = get_matching_rewards(
            entries=[entries[i] for i in matching_indices],
            pred_responses=[entries[i][model_output_key] for i in matching_indices],
        )
        rewards[matching_indices] = matching_rewards
    
    # TODO: support LLM-as-judge in the future
    
    

    return {
        "rewards": rewards.tolist(),
        "correctness_rewards": rewards.tolist()
    }

def is_legal_cot_split_boxed(s):
    # Required substrings in order
    # First, let's check the main tags are in order and appear once
    required_tags = ['<Thought>', '</Thought>', '<Output>', '</Output>']
    last_pos = -1
    
    # Check order and uniqueness of main tags
    for tag in required_tags:
        pos = s.find(tag)
        
        # Check if tag exists
        if pos == -1:
            return False, f"Missing tag: {tag}"
        
        # Check if it appears after the previous tag
        if pos <= last_pos:
            return False, f"Incorrect order: {tag} should appear after previous tag"
        
        # Check if tag appears more than once
        if s.count(tag) > 1:
            return False, f"Tag appears multiple times: {tag}"
        
        last_pos = pos

    # Now, let's check the 'boxed' requirement within Output tags
    # Extract content between Output tags
    output_start = s.find('<Output>') + len('<Output>')
    output_end = s.find('</Output>')
    output_content = s[output_start:output_end]
    boxed_count = output_content.count('boxed')
    if boxed_count == 0:
        return False, "'boxed' is missing within Output tags"
    elif boxed_count > 1:
        return False, "'boxed' appears multiple times within Output tags"
    
    return True, "String is legal"


def compute_score(
        solution_str, 
        ground_truth,
        extra_info,
        pattern_mode: Literal['boxed', 'strict_boxed', 'cot_split_boxed'] = 'cot_split_boxed',
        boxed_reward=True, 
        use_ray=False,
        print_verifier_example=False,
        ) -> List[Dict]:

    extra_info = deepcopy(extra_info)
    extra_info['cases'] = json.loads(extra_info['cases'])
    
    if extra_info['original_source'] == 'mcq_26m' and extra_info['answer_type'] in ['a', 'b', 'c', 'd', 'e']:
        data_type = ('mcq_26m', extra_info['answer_type'])
    elif extra_info['original_source'] == 'mcq_26m_close_form' and extra_info['answer_type'] in ['a', 'b', 'c']:
        data_type = ('mcq_26m_close_form', extra_info['answer_type'])
    else:
        data_type = (extra_info['Topic'], extra_info['answer_type'])

    math_indices = [('Math', 'a'), ('Math', 'b'), ('Reasoning', 'a'), ('mcq_26m_close_form', 'a'), ('mcq_26m_close_form', 'b')]
    code_indices = [('Code', 'assert'), ('Code', 'input')]
    mcq_indices = [('mcq_26m', 'a'), ('mcq_26m', 'b'), ('mcq_26m', 'c'), ('mcq_26m', 'd'), ('mcq_26m', 'e')]
    matching_indices = [('Reasoning', 'a')]
    
    if data_type in math_indices:
        correctness_reward = get_math_rewards(
            ref_responses=[ground_truth],
            pred_responses=[solution_str],
            enable_math_expr_extract=True,
            use_ray=use_ray
        )[0]
    elif data_type in code_indices:
        correctness_reward = get_code_rewards(
            entries=[extra_info],
            pred_responses=[solution_str],
            use_ray=use_ray
        )[0]
    elif data_type in mcq_indices:
        correctness_reward = get_mcq_rewards(
            entries=[extra_info],
            pred_responses=[solution_str]
        )[0]
    elif data_type in matching_indices:
        correctness_reward = get_matching_rewards(
            entries=[extra_info],
            pred_responses=[solution_str]
        )[0]
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    # reset the rewards
    if boxed_reward:
        # result['is_boxed'] = []
        for i in range(len(result['rewards'])):
            
            # strict boxed reward
            if pattern_mode == 'strict_boxed':
                if pred_responses[i].count('boxed')==1:
                    result['is_boxed'].append(1)
                    if result['correctness_rewards'][i] > 0.0:
                        result['rewards'][i] = 1.0
                    else:
                        result['rewards'][i] = -0.5
                else:
                    result['is_boxed'].append(0)
                    result['rewards'][i] = -1.0
            # simpleRL reward
            elif pattern_mode == 'boxed':
                if result['correctness_rewards'][i] > 0.0:
                    result['is_boxed'].append(1)
                else:
                    if 'boxed' not in pred_responses[i]:
                        result['rewards'][i] = -1.0
                        result['is_boxed'].append(0)
                    else:
                        result['rewards'][i] = -0.5
                        result['is_boxed'].append(1)
            elif pattern_mode == 'cot_split_boxed':
                is_legal, illegal_reason = is_legal_cot_split_boxed(pred_responses[i])
                result ['is_boxed'].append(int(is_legal))
                if result['correctness_rewards'][i] > 0.0 and is_legal:
                    result['rewards'][i] = 1.0
                # elif result['correctness_rewards'][i] > 0.0 and not is_legal:
                #     result['rewards'][i] = 0.0
                elif result['correctness_rewards'][i] <= 0.0 and is_legal:
                    result['rewards'][i] = -0.5
                else:
                    result['rewards'][i] = -1.0
            else:
                raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

    return result


