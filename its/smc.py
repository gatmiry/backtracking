import argparse
import os
import json
from vllm import LLM, SamplingParams
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import re
import torch.nn.functional as F
import numpy as np

import regex

from glob import glob

from datasets import load_dataset
from collections import defaultdict
import pickle
import cvxpy as cp

from math_verify import verify, parse


import transformers.cache_utils as cache_utils

# Check if the method is missing and patch it
if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
    def get_usable_length(self, seq_len=None, layer_idx=0):
        # In newer versions, get_seq_length performs the same role
        return self.get_seq_length(layer_idx)
    
    cache_utils.DynamicCache.get_usable_length = get_usable_length
    print("Successfully patched DynamicCache.get_usable_length")

def constrained_solver(r,  eps, q = 1, lmbda = 0.4):
    
    r = np.asarray(r)
    ## transform it into logits
    r = inverse_sigmoid(r)
    
    N = r.size

    s = cp.Variable(N)

    obj = lmbda * cp.log_sum_exp(s / lmbda)
    
    cons = [
        cp.sum(s) == np.sum(r),
        cp.norm(s - r, q) <= eps
    ]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCS) # or solver='ECOS', 'MOSEK', etc.

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver status: {prob.status}")

    s_opt = s.value
    ex = np.exp(s_opt / lmbda)
    p_opt = ex / ex.sum()

    return p_opt


def inverse_sigmoid(x):
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1 - eps)

    return np.log(x) - np.log(1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    
    return exp_x / exp_x.sum()

def sampling_function(rewards, softmax_temp = 0.8):
    logits = [inverse_sigmoid(r) for r in rewards]
    logits = np.array(logits)
    weights = softmax(logits / softmax_temp)
    
    return weights

def smc(answers, reward_new, reward_old, softmax_temp = 0.8):
    score_new = [inverse_sigmoid(r) for r in reward_new]
    score_old = [inverse_sigmoid(r) for r in reward_old]
    # logits_ratio = [n / o for n, o in zip(score_new, score_old)]
    logits_diff = [n - o for n, o in zip(score_new, score_old)]
    logits_diff = np.array(logits_diff)
    weights = softmax(logits_diff / softmax_temp)
    
    return np.random.choice(answers, size=len(reward_new), replace=True, p=weights)



def verify(response, truth):
    
    pattern = r'\\boxed{((?:[^{}]|{(?1)})*)}'

    try:
        ans = regex.findall(pattern, response)[0]
    except Exception as e:
        return False

    return math_equal(memoized_canonical_form(ans), memoized_canonical_form(truth))
    

def load_dataset(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as infile:
            for line in infile:                
                entry = json.loads(line)
                
                data.append(entry)

    except :
        print("error")
        
    return data

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


BOXED_PATTERN = r'\\boxed{((?:[^{}]|{(?1)})*)}'

def extract_boxed_answer(response):
    try:
        ans = regex.findall(BOXED_PATTERN, response)[-1]
        return ans
    except Exception:
        return None # No boxed answer found

def verify(model_response, true_answer):

    try:
        result = verify(parse(model_response), parse(true_answer))
        return result
    except Exception:
        return False
    

def load_dataset(filename):
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as infile:
            for line in infile:                
                entry = json.loads(line)
                
                data.append(entry)

    except :
        print("error")
        
    return data


"""Math answer verification using math_verify library (Hugging Face standard)."""

import re
from math_verify import parse, verify, LatexExtractionConfig, StringExtractionConfig, ExprExtractionConfig


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} using proper brace matching."""
    if not text:
        return None
        
    matches = list(re.finditer(r'\\box(ed)?\{', text))
    if not matches:
        return None
    
    start_pos = matches[-1].end()
    depth = 1
    i = start_pos
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start_pos:i-1].strip()
    return None


def normalize_latex_for_parsing(s: str) -> str:
    """Normalize LaTeX commands before parsing."""
    if not s:
        return ""
    # Remove variable assignment prefixes (M=, V=, etc.)
    s = re.sub(r'^[A-Za-z]\s*=\s*', '', s)
    # Remove currency symbols
    s = s.replace('\\$', '').replace('$', '')
    # Normalize exponent braces: ^{1234} -> ^1234 (for numeric exponents only)
    # This helps math_verify parse exponents correctly
    s = re.sub(r'\^\{(\d+)\}', r'^\1', s)
    # Normalize fraction variants
    s = s.replace('\\dfrac', '\\frac')
    s = s.replace('\\tfrac', '\\frac')
    # Fix \frac followed by space and single char/digit: \frac 9{...} -> \frac{9}{...}
    s = re.sub(r'\\frac\s+(\d|\w)\s*\{', r'\\frac{\1}{', s)
    # Fix \sqrt followed by space and single char/digit: \sqrt 2 -> \sqrt{2}
    s = re.sub(r'\\sqrt\s+(\d|\w)(?![a-zA-Z0-9])', r'\\sqrt{\1}', s)
    # Normalize spacing commands
    s = s.replace('\\ ', ' ')
    s = s.replace('\\,', ' ')
    s = s.replace('\\;', ' ')
    s = s.replace('\\:', ' ')
    s = s.replace('\\!', '')
    s = s.replace('\\quad', ' ')
    s = s.replace('\\qquad', ' ')
    # Remove display math delimiters
    s = re.sub(r'\\\[|\\\]|\\\(|\\\)', '', s)
    # Normalize trig function parentheses: \sin^2(x) -> \sin^2 x
    s = re.sub(r'\\(sin|cos|tan|cot|sec|csc|log|ln)\^(\{[^{}]+\}|\d+)\(([^)]+)\)', r'\\\1^\2 \3', s)
    # Normalize \left\lceil and \right\rceil to plain \lceil \rceil
    s = s.replace('\\left\\lceil', '\\lceil').replace('\\right\\rceil', '\\rceil')
    s = s.replace('\\left\\lfloor', '\\lfloor').replace('\\right\\rfloor', '\\rfloor')
    return s


def normalize_for_comparison(s: str) -> str:
    """Aggressively normalize for string comparison."""
    if not s:
        return ""
    s = normalize_latex_for_parsing(s)
    # Convert \frac{a}{b} to a/b for comparison
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    # Remove all spaces
    s = re.sub(r'\s+', '', s)
    # Normalize degree symbol
    s = s.replace('^\\circ', '°').replace('^{\\circ}', '°').replace('\\circ', '°')
    # Remove parentheses around single arguments
    s = re.sub(r'\((\d+°?)\)', r'\1', s)
    # Simplify (1)/(x) to 1/x, 1/(x) to 1/x
    s = re.sub(r'\((\d+)\)/\(([^()]+)\)', r'\1/\2', s)
    s = re.sub(r'\(([^()]+)\)/\((\d+)\)', r'\1/\2', s)
    s = re.sub(r'(\d+)/\(([^()]+)\)', r'\1/\2', s)
    s = re.sub(r'\(([^()]+)\)/(\d+)', r'\1/\2', s)
    # Normalize operators
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    # Lowercase
    s = s.lower()
    return s


def normalize_text(s: str) -> str:
    """Normalize text for comparison (handles \\text{}, whitespace, case)."""
    if not s:
        return ""
    s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)  # Remove LaTeX commands
    s = re.sub(r'[{}\[\]$]', '', s)     # Remove braces
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


def check_answer(generated: str, ground_truth: str) -> bool:
    """
    Check if generated answer matches ground truth using math_verify.
    
    Args:
        generated: Full generated text (should contain \\boxed{...})
        ground_truth: The correct answer string
        
    Returns:
        True if answers match, False otherwise
    """
    # Extract boxed answer from generated text
    gen_ans = extract_boxed_answer(generated)
    if gen_ans is None:
        return False
    
    # Normalize LaTeX before parsing
    gt_normalized = normalize_latex_for_parsing(ground_truth)
    gen_normalized = normalize_latex_for_parsing(gen_ans)
    
    # FIRST: Check for tuple lists - these need special handling
    # because math_verify incorrectly extracts single numbers from tuple lists
    gt_tuples = extract_tuples(ground_truth)
    gen_tuples = extract_tuples(gen_ans)
    if gt_tuples and len(gt_tuples) >= 2:
        # Ground truth is a tuple list - use tuple comparison
        if gen_tuples and gt_tuples == gen_tuples:
            return True
        # If tuples exist but don't match, don't fall through to math_verify
        # (which would incorrectly match based on individual numbers)
        if gen_tuples:
            return False
    
    # Use latex, expr, and string extraction for comprehensive parsing
    config = [
        LatexExtractionConfig(boxed_match_priority=100),
        ExprExtractionConfig(),
        StringExtractionConfig()
    ]
    
    try:
        # Try parsing with normalized strings
        gt_parsed = parse(gt_normalized, extraction_config=config)
        gen_parsed = parse(gen_normalized, extraction_config=config)
        
        # If both parsed successfully, use math_verify
        if gt_parsed and gen_parsed:
            result = verify(gt_parsed[0], gen_parsed[0], strict=False)
            if result:
                return True
            
            # Try sympy simplification for exponential equivalence (e.g., 4^{2006} = 2^{4012})
            try:
                from sympy import simplify, Eq
                diff = simplify(gt_parsed[0] - gen_parsed[0])
                if diff == 0:
                    return True
                # Also try checking equality directly
                if simplify(Eq(gt_parsed[0], gen_parsed[0])) == True:
                    return True
            except:
                pass
        
        # If only one parsed, try numeric comparison
        if gt_parsed and not gen_parsed:
            try:
                gen_val = float(gen_ans.replace(',', ''))
                from sympy import N
                gt_val = float(N(gt_parsed[0]))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
                # Check relative error for large numbers
                if gt_val != 0 and abs((gt_val - gen_val) / gt_val) < 1e-9:
                    return True
            except:
                pass
        
        if gen_parsed and not gt_parsed:
            try:
                gt_val = float(ground_truth.replace(',', ''))
                from sympy import N
                gen_val = float(N(gen_parsed[0]))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
                if gen_val != 0 and abs((gt_val - gen_val) / gen_val) < 1e-9:
                    return True
            except:
                pass
        
        # Try both numeric if both failed to parse
        if not gt_parsed and not gen_parsed:
            try:
                gt_val = float(ground_truth.replace(',', ''))
                gen_val = float(gen_ans.replace(',', ''))
                if abs(gt_val - gen_val) < 1e-6:
                    return True
            except:
                pass
        
        # Fallback for text answers that don't parse (e.g., "Yes", "No")
        gt_clean = normalize_text(ground_truth)
        gen_clean = normalize_text(gen_ans)
        
        if gt_clean and gen_clean and gt_clean == gen_clean:
            return True
            
    except Exception:
        pass



# Fallback: tuple set comparison (order-independent)
    gt_tuples = extract_tuples(ground_truth)
    gen_tuples = extract_tuples(gen_ans)
    if gt_tuples and gen_tuples and gt_tuples == gen_tuples:
        return True
    
    # Fallback: aggressive normalization (removes all spaces)
    gt_aggr = normalize_for_comparison(ground_truth)
    gen_aggr = normalize_for_comparison(gen_ans)
    if gt_aggr and gen_aggr and gt_aggr == gen_aggr:
        return True
    
    # DISABLED: latex2sympy2_extended fallback is too slow
    # The math_verify library already handles most cases
    
    # Final fallback: normalized string comparison
    return normalize_text(ground_truth) == normalize_text(gen_ans)


def extract_tuples(text: str) -> set | None:
    """Extract all tuples from text as a set (order-independent comparison)."""
    if not text:
        return None
    # Find all (a, b, ...) patterns
    tuples = re.findall(r'\(([^()]+)\)', text)
    if not tuples:
        return None
    result = set()
    for t in tuples:
        # Split by comma, normalize each element
        parts = tuple([p.strip().replace(' ', '').replace('\\', '').lower() for p in t.split(',')])
        if len(parts) >= 2:
            result.add(parts)
    return result if result else None

        

def main():
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")

    parser.add_argument('--method', type=str, required=True, choices=['best-of-n', 'pf', 'maxmin'],
                        help='The method to use for the experiment.')
    parser.add_argument('--q', type=float, required=True,
                        help='A numerical parameter q.')
    parser.add_argument('--eps', type=float, required=True,
                        help='A numerical parameter epsilon (eps).')
    parser.add_argument('--N', type=int, required=True,
                        help='A numerical parameter N.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output file for saving results.')

    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Path to the main model.')
    parser.add_argument('--prm', type=str, default='Qwen/Qwen2.5-Math-PRM-7B',
                        help='Path to the PRM model.')
    parser.add_argument('--dataset_path', type=str, default='../datasets/math500_100randomquestions.jsonl',
                        help='Path to the dataset file.')
    parser.add_argument('--max_step', type=int, default=10,
                        help='Maximum number of steps.')

    args = parser.parse_args()

    method = args.method
    model_path = args.model
    prm_path = args.prm
    dataset_path = args.dataset_path
    output_file = args.output_file
    q=args.q
    eps=args.eps
    N=args.N
    max_step=args.max_step


    my_dataset = load_dataset(dataset_path)
    
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        seed=96,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=4096,
    )    
        
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=2048,
        top_p=1.0,
        stop=["\n\n", "<|eot_id|>","\n##"],
    )
    
    tokenizer = llm.get_tokenizer()
    system = [
                {
                "role": "system",
               "content": "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.",
                }
             ]
    
    prm = AutoModel.from_pretrained(prm_path,
                                            device_map="cuda:1",
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True).eval()
    
    prm_tokenizer = AutoTokenizer.from_pretrained(prm_path)
    
    # try:
    with open(output_file, 'w', encoding='utf-8') as outfile:

        for data in my_dataset:
            
        
            question = data['problem']
            true_answer = data['answer']
            difficulty = data['level']
            
            
            prompt = tokenizer.apply_chat_template(
                                                    system + [{"role": "user", "content": question}],
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                  )
            ## generating N different full answers
            
            generated_answers = []
            for n in range(N):
                res = llm.generate(prompt, sampling_params)
                generated_answers.append(res[0].outputs[0].text)        
        
            prm_scores = []
            
            for answ in generated_answers:
                
                ans = re.sub(r'\n+', '\n', answ)
                steps_list = ans.split("\n\n")
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                ]

                conversation = prm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                    
                input_ids = prm_tokenizer.encode(
                    conversation, 
                    return_tensors="pt", 
                ).to(prm.device)
                
                outputs = prm(input_ids=input_ids)    
                
                step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                prm_scores.append(step_reward[0][0])

            ## todo: add an if statement to change the sampling algorithm
            ### for the first iteration, smc and pg are identical, as prior scores are the same
            weights = sampling_function(prm_scores) 
            generated_answers_resampled = np.random.choice(generated_answers, size=N, replace=True, p=weights)

            #weights = constrained_solver(prm_scores, eps, q)
            # generated_answers_resampled = generated_answers.copy() ## bon


            ### this is the part needs to be changed to have smc instead, the main difference is backtracking and prob 
            ### okay actually seems like smc doesnt have backtracking, the only difference is that it requires ratios
            ### ratios of the new rewards to the old rewards

            ### for the first iteration we dont need to calculate the ratios

            final_set = []
            step = 0
            while len(generated_answers_resampled) > 0 and step <= max_step:
                step += 1
                prompt = tokenizer.apply_chat_template(
                                                        system + [{"role": "user", "content": question}],
                                                        tokenize=False,
                                                        add_generation_prompt=True,
                                      )
                
                generated_answers = []
                prm_scores_old = prm_scores.copy() ## added for smc implementation
                prm_scores = []
                is_active = []
                for answ in generated_answers_resampled:
                
                
                    prompt_answ = prompt + "\n\n" + answ + "\n\n" ## should i force it to take the step by adding "Step" to the prompt?
                    res = llm.generate(prompt_answ, sampling_params)
                    if res and res[0].outputs:
                        generated_answers.append(answ + "\n\n" + res[0].outputs[0].text)
                        ## todo implement sth if the generation is over
                        if tokenizer.eos_token_id in res[0].outputs[0].token_ids:
                            is_active.append(False)
                            final_set.append(generated_answers[-1])
                        else:
                            is_active.append(True)
                
                
                        ans = re.sub(r'\n+', '\n', generated_answers[-1])
                        steps_list = ans.split("\n\n")
                        messages = [
                            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                        ]
                            
                        conversation = prm_tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                            
                        input_ids = prm_tokenizer.encode(
                            conversation, 
                            return_tensors="pt", 
                        ).to(prm.device)
                        
                        outputs = prm(input_ids=input_ids)    
                        
                        step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
                        token_masks = (input_ids == step_sep_id)
                        step_reward = make_step_rewards(outputs[0], token_masks)
                        prm_scores.append(step_reward[0][0])

                    else:
                        
                        print("-----------")
                        print("WARNING: LLM generated an empty output.")
                        print("Prompt was:", prompt_answ)
                        print("Full response object:", res)
                        print("-----------")



                
                filtered_prm = [sc for sc, act in zip(prm_scores, is_active) if act]
                filtered_prm_old = [sc for sc, act in zip(prm_scores_old, is_active) if act] ### needed for smc

                
                if len(filtered_prm) > 0:
                    #weights = sampling_function(filtered_prm)
                    #weights = constrained_solver(filtered_prm, eps, q)
                    filtered_answers = [answ for answ, act in zip(generated_answers, is_active) if act]
                    #generated_answers_resampled = np.random.choice(filtered_answers, size=len(filtered_prm), replace=True, p=weights)
                    # generated_answers_resampled = filtered_answers.copy() ## bo
                    generated_answers_resampled = smc(filtered_answers, filtered_prm, filtered_prm_old)
                    ## len(generated_answers[is_active]) later needs to be changed to the width of the sampling
                else:
                    generated_answers_resampled = []
            
            ## force the model to output answers if it didn't reach the answer within the limits. but for now let's just print
            if len(generated_answers_resampled) > 0:
                for answ in generated_answers_resampled:
                    prompt_answ = prompt + answ + "\n\nTherefore, the final answer is \\boxed{"
                    res = llm.generate(prompt_answ, sampling_params)
                    final_set.append(answ + "\n\nTherefore, the final answer is \\boxed{" + res[0].outputs[0].text)
        
        
            res = []
            all_rew = []
            for ans in final_set:
                res.append(check_answer(ans, true_answer))
                steps_list = ans.split("\n\n")
        
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                ]
            
                conversation = prm_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            
                input_ids = prm_tokenizer.encode(
                    conversation, 
                    return_tensors="pt", 
                ).to(prm.device)
            
                outputs = prm(input_ids=input_ids)
            
            
                step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                all_rew.append(step_reward[0][-1])
                                
        
            output_record = {
                'question': question,
                'true_answer': true_answer,
                'answers': final_set,
                'difficulty_level': difficulty,
                'prm': all_rew,
                'result': res,
            }
        
            outfile.write(json.dumps(output_record) + '\n')
            
    # except Exception as e:
    #     print(f"An error occurred: {e}")




if __name__ == '__main__':
    main()





