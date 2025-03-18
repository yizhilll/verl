import json
import ray
import numpy as np
from copy import deepcopy
import os
from typing import TypeVar, Dict, List, Union, Any, Tuple, Literal
import re
import signal


# from evaluation.parser import extract_answer, find_box
from .code import extract_code_and_test
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
# from openrlhf.utils.data_record_util import JsonlFileHandler


MAX_INS_LEN_AS_KEY = 2400

def extract_label_content(text: str) -> Tuple[bool, str, str]:
    """
    Check if the string looks like a label and extract content after it.
    
    Args:
        text (str): Input text to check
    
    Returns:
        tuple[bool, str, str]: (is_label, label, content)
            - is_label: Whether the text matches a label pattern
            - label: The label part if found, empty string if not
            - content: The content after label if found, empty string if not
    """
    text = text.strip()
    patterns = [
        # pattern, group names for (label, content)
        (r'^([A-Z]\.)\s*(.+)?$', 1, 2),         # Matches "A.", "B. sin(x)"
        (r'^([A-Z]\d+\.)\s*(.+)?$', 1, 2),      # Matches "Q1.", "P2. anything"
        (r'^(\d+\.)\s*(.+)?$', 1, 2),           # Matches "1.", "42. anything"
        (r'^([A-Z]\s*\d+)\s*(.+)?$', 1, 2),     # Matches "C 7", "A 42 anything"
        (r'^\(([A-Z])\)\s*(.+)?$', 1, 2),       # Matches "(A)", "(B) content"
        (r'^([A-Z]\))\s*(.+)?$', 1, 2),         # Matches "A)", "B) content"
        (r'^\[([A-Z])\]\s*(.+)?$', 1, 2),       # Matches "[A]", "[B] content"
    ]
    
    for pattern, label_group, content_group in patterns:
        match = re.match(pattern, text)
        if match:
            label = match.group(label_group).strip()
            # Remove any trailing parenthesis or bracket from the label
            label = re.sub(r'[\)\]]$', '', label)
            content = match.group(content_group).strip() if match.group(content_group) else ""
            return True, label, content
            
    return False, "", ""


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Process timeout")

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
    
    # import logging

    # # Store the original level
    # original_level = logging.getLogger("math_verify.utils").level

    # # Set the logger to a higher level (ERROR or CRITICAL) before calling the function
    # logging.getLogger("math_verify.utils").setLevel(logging.CRITICAL)


    if len(gold_parsed) != 0:
        return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
    else:
        gold_parsed = parse(f'${gold_standard}$', extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            return float(verify(gold_parsed, answer_parsed) or verify(gold_parsed, answer_string_parsed))
        else:
            assert ValueError, f"Invalid gold standard: {gold_standard}, which should be filtered before training"
    
    
    # # Restore the original logger level
    # logging.getLogger("math_verify.utils").setLevel(original_level)
    
    return 0.0


# 修改 safe_rule_based_verifier 函数
# def safe_rule_based_verifier(
#     output: str, 
#     gold_standard: str, 
#     enable_math_expr_extract: bool,
#     retry: int = 2,
#     timeout_per_try: int = 3,
# ) -> float:
#     current_try = 0
#     while current_try < retry:
#         try:
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(timeout_per_try) # `timeout_per_try` must be integer
#             reward = rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)
#             signal.alarm(0)
#             return reward
#         except Exception as e:
#             print(f'Trying rule_based_verifier() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
#             print(f'error output:', e)
#             current_try += 1
    
#     print(f"[WARNING]: Rule-based verifier failed after {retry} attempts with timeout")
#     return 0.0

def timeout(timeout_seconds: int = 8):
    if os.name == "posix":
        import signal

        def decorator(func):

            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator
    else:
        raise NotImplementedError(f"Unsupported OS: {os.name}")

@timeout(timeout_seconds=10)
def safe_rule_based_verifier(
    output: str, 
    gold_standard: str, 
    enable_math_expr_extract: bool,
    retry: int = 2,
    timeout_per_try: int = 3,
) -> float:
    
    try:
        reward = rule_based_verifier(output=output, gold_standard=gold_standard, enable_math_expr_extract=enable_math_expr_extract)
        return reward
    except Exception as e:
        print(f'[WARNING]: Trying rule_based_verifier() {current_try+1}: running over {timeout_per_try} seconds, killed by timeout error')
        print(f'error output:', e)
    
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
    
def get_matching_rewards(ground_truths: List[Dict], pred_responses: List[str]) -> List[float]:
    rewards = []
    
    for idx, (ground_truth, pred) in enumerate(zip(ground_truths, pred_responses)):
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
        # print_verifier_example=False,
        ) -> float:

    # print(f"##### get score for #####\nsolution:{solution_str}\nground_truth:{ground_truth}")
    if pattern_mode == 'cot_split_boxed':
        is_legal, illegal_reason = is_legal_cot_split_boxed(solution_str)
        if not is_legal:
            return -1.0

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
        # correctness_reward = get_mcq_rewards(
        #     entries=[extra_info],
        #     pred_responses=[solution_str]
        # )[0]
        raise NotImplementedError("MCQ reward calculation is not supported")
    elif data_type in matching_indices:
        correctness_reward = get_matching_rewards(
            ground_truths=[ground_truth],
            pred_responses=[solution_str]
        )[0]
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    is_boxed = 0 # whether the solution is well-formatted
    
    # reset the rewards
    if boxed_reward:
        # strict boxed reward
        if pattern_mode == 'strict_boxed':
            if solution_str.count('boxed')==1:
                is_boxed = 1
                if correctness_reward > 0.0:
                    final_score = 1.0
                else:
                    final_score = -0.5
            else:
                is_boxed = 0
                final_score = -1.0
        # simpleRL reward
        elif pattern_mode == 'boxed':
            if correctness_reward > 0.0:
                is_boxed = 1
            else:
                if 'boxed' not in solution_str:
                    final_score = -1.0
                    is_boxed = 0
                else:
                    final_score = -0.5
                    is_boxed = 1
        elif pattern_mode == 'cot_split_boxed':
            is_legal, illegal_reason = is_legal_cot_split_boxed(solution_str)
            is_boxed = int(is_legal)
            if correctness_reward > 0.0 and is_legal:
                final_score = 1.0
            # elif correctness_reward > 0.0 and not is_legal:
            #     final_score = 0.0
            elif correctness_reward <= 0.0 and is_legal:
                final_score = -0.5
            else:
                final_score = -1.0
        else:
            raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

    return final_score