from typing import List, Sequence, Dict, Any
import sglang as sgl
import classifier_lib as classifier_lib
import torch
import random

def score_generation(model_input, classifier, device, dtype):
     # now score each generation
    #print('model input is: ', model_input)
    input_ids = torch.tensor(model_input, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
    with torch.autocast(device_type=device.type, dtype=dtype):
        classifier_outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
        score = classifier_outputs.success_probs[0, -1].item()
    
    return score

def generate_backtrack_batch(
    piref_model: sgl.Engine,
    prompt_ids: Sequence[List[int]],
    max_length: int,
    block_size: int,
    max_blocks: int,
    temperature: float,
    top_p: float,
    stop_token_ids: Sequence[int],
    max_value_estimate_num_attempts: int,
    classifier_model: classifier_lib.Qwen2ForClassifier,
    use_rejection_sampling: bool
) -> List[Dict[str, Any]]:

    device = classifier_model.device
    device_type = device.type
    block_history: List[List[List[int]]] = [[] for _ in range(len(prompt_ids))]
    current_tokens_list: List[List[int]] = [[] for _ in range(len(prompt_ids))]
    ## the legnth of block_kept_count is not going to change, same with current_scores
    blocks_kept_count = [0 for _ in range(len(prompt_ids))]
    #current_scores = [0.0 for _ in range(len(prompt_ids))]
    ## the legnth of remaining_indices is going to change
    max_attempts = max_blocks * 16
    remaining_indices = list(range(len(prompt_ids)))
    attempts = 0
    while len(remaining_indices) > 0 and attempts < max_attempts:
        tmp_remaining_indices = []
        print(f"Attempt {attempts}. length of remaining indices: {len(remaining_indices)}")
        attempts += 1
        
        sampling_params_list = []
        model_input_list = []
        for remaining_index in remaining_indices:
            print(f"remaining index: {remaining_index}")
            current_tokens = current_tokens_list[remaining_index]
            remaining_tokens_count = max_length - len(current_tokens)
            if remaining_tokens_count <= 0 or blocks_kept_count[remaining_index] >= max_blocks:
                continue
        ## we will update current_tokens_list and blocks_kept_count at the end
            tmp_remaining_indices.append(remaining_index)

            ## now make the sampling params and add it to sampling_params_list
            max_new_tokens = min(block_size, remaining_tokens_count)
            sampling_params = {
                "temperature": temperature,
                "top_p": top_p,
                "skip_special_tokens": False,
                "stop_token_ids": list(stop_token_ids),
                "max_new_tokens": max_new_tokens,
            }
            sampling_params_list.append(sampling_params)

            model_input = prompt_ids[remaining_index] + current_tokens
            model_input_list.append(model_input)

        remaining_indices = tmp_remaining_indices             
        if len(remaining_indices) == 0:
            break

        average_scores = [0.0 for _ in range(len(remaining_indices))]
        max_scores = [-float('inf') for _ in range(len(remaining_indices))]
        current_max_generations = [[] for _ in range(len(remaining_indices))]
        current_scores = []
        for i in range(len(model_input_list)):
            #print('model input list is: ', model_input_list[i])
            current_score = score_generation(model_input_list[i], classifier_model, device, torch.bfloat16)
            current_scores.append(current_score)

        for _ in range(max_value_estimate_num_attempts):
            print(f"max value estimate num attempts: {_}")
            infer_outputs = piref_model.generate(input_ids=model_input_list, sampling_params=sampling_params_list)
            for i, infer_output in enumerate(infer_outputs):
                print(f"infer output: {infer_output}")
                tmp_input_ids = model_input_list[i].copy()
                tmp_input_ids.extend(infer_output["output_ids"])
                print(f'entered score generation for index {i}')
                score = score_generation(tmp_input_ids, classifier_model, device, torch.bfloat16)
                print(f'score: {score}')
                average_scores[i] += score
                if score > max_scores[i]:
                    max_scores[i] = score
                    current_max_generations[i] = infer_output["output_ids"]
        average_scores = [score / max_value_estimate_num_attempts for score in average_scores]
        for i, remaining_index in enumerate(remaining_indices):
            backtrack_prob = current_scores[i] / (current_scores[i] + average_scores[i] + 1e-10)
            if random.random() >= backtrack_prob or len(block_history[remaining_index]) == 0:
                current_tokens_list[remaining_index].extend(current_max_generations[i])
                blocks_kept_count[remaining_index] += 1
                block_history[remaining_index].append(current_max_generations[i])
                print(f'length of current_max_generations[i]: {len(current_max_generations[i])}')
            else:
                last_block = block_history[remaining_index].pop()
                del current_tokens_list[remaining_index][-len(last_block):]
                blocks_kept_count[remaining_index] -= 1
    
    # Convert to the format expected by the caller (list of dicts with 'output_ids' key)
    return [{"output_ids": tokens} for tokens in current_tokens_list]
