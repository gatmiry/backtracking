import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence
import classifier_lib
import numpy as np

import torch
import transformers
import tyro
import ujson as json
import sglang as sgl
from torch.utils.data import DataLoader, Dataset

import benchmark_data
import deepseek_utils
import accuracy_utils
import eval_helpers

import wandb
from tqdm import tqdm

@dataclass
class Args:
    benchmark: str = "aime-24"
    max_samples: int = 16
    seed: int = 1337
    piref_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    output_path: str = "outputs/inference_outputs.jsonl"
    attention_impl: str = "sdpa"
    piref_gpu_util: float = 0.5
    batch_size: int = 1
    max_length: int = 8192 ##16384
    block_size: int = 4096 ##4096
    max_blocks: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    classifier_ckpt_path: str = "VGS-AI/DeepSeek-VM-1.5B"
    use_rejection_sampling: bool = True
    max_value_estimate_num_attempts: int = 4
    num_repetitions: int = 1 
    dataset_size: int = -1
    wandb_project: str = "PRM_prediction_AME24_backtrack"
    gpu_id: int = 7

    def __post_init__(self) -> None:
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


def get_eos_token_id(tokenizer: transformers.PreTrainedTokenizer) -> int:
    eos_token_id = tokenizer.convert_tokens_to_ids(["<｜end▁of▁sentence｜>"])[0]
    return eos_token_id


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        example = dict(self.base_dataset[base_idx])
        example["data_idx"] = base_idx
        return example


def score_all_generations(classifier_model, prompt_ids, generation_ids, device_type, dtype):
     # now score each generation
    input_output_ids = []
    num_prompts = len(prompt_ids)
    num_responses = len(generation_ids[0])
    for i in range(num_prompts):
        assert len(generation_ids[i]) == num_responses
        for j in range(num_responses):
            input_output_ids.append(prompt_ids[i] + generation_ids[i][j])

    # since we give lots of memory to sgl, the classifier might oom here. thus, we for loop to save memory
    scores_list = []
    device = classifier_model.device
    for i in tqdm(range(num_prompts * num_responses), desc="Scoring generations"):
        input_ids = torch.tensor(input_output_ids[i], device=device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
        with torch.autocast(device_type=device_type, dtype=dtype):
            classifier_outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            scores = classifier_outputs.success_probs[:, -1].item()
        scores_list.append(scores)
    scores = torch.tensor(scores_list).view(num_prompts, num_responses).tolist()
    return scores


def generate_single_line(
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
    use_rejection_sampling: bool,
) -> List[List[int]]:
   
    stop_token_set = set(stop_token_ids)
    outputs: List[List[int]] = []

    for prompt in prompt_ids:
        current_tokens: List[int] = []
        block_history: List[List[int]] = []
        blocks_kept = 0
        attempts = 0
        max_attempts = max(max_blocks * 4, max_blocks + 1)

        print('prompt is ', prompt)
        current_reward = 0
        while blocks_kept < max_blocks and len(current_tokens) < max_length and attempts < max_attempts:
            print(f"Attempt {attempts}. length of current tokens: {len(current_tokens)}")
            remaining_tokens = max_length - len(current_tokens)
            if remaining_tokens <= 0:
                break

            attempts += 1
            max_new_tokens = min(block_size, remaining_tokens)
            sampling_params = [{
                "temperature": temperature,
                "top_p": top_p,
                "skip_special_tokens": False,
                "stop_token_ids": list(stop_token_ids),
                "max_new_tokens": max_new_tokens,
            }]

            model_input = prompt + current_tokens

            ## estimate the maximum of the rewards
            max_reward = 0
            average_reward = 0
            max_reward_block_tokens = None

            ## estimate the maximum of the rewards using rejection sampling, also the average reward
            for _ in range(max_value_estimate_num_attempts):
                tmp_input_ids = model_input.copy()
                infer_outputs = piref_model.generate(input_ids=[model_input], sampling_params=sampling_params)
                block_tokens = infer_outputs[0]["output_ids"]
                if not block_tokens:
                    break
                if len(block_tokens) > max_new_tokens:
                    block_tokens = block_tokens[:max_new_tokens]
                tmp_input_ids.extend(block_tokens)
                input_ids = torch.tensor(tmp_input_ids, device=classifier_model.device, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones(input_ids.shape, device=classifier_model.device, dtype=torch.long)
                reward_sample = classifier_model(input_ids=input_ids, attention_mask=attention_mask).success_probs[0, -1].item()
                average_reward += reward_sample
                if reward_sample > max_reward:
                    max_reward = reward_sample
                    max_reward_block_tokens = block_tokens
            average_reward /= max_value_estimate_num_attempts
            backtrack_prob = current_reward / (average_reward + current_reward)
            if random.random() < backtrack_prob:
                    print(f"Backtracking at attempt {attempts}.")
                    removed_block = block_history.pop()
                    del current_tokens[-len(removed_block):]
                    print(f"Removed block: {removed_block}")
                    blocks_kept -= 1
                    continue
            else:
                print(f"Not backtracking at attempt {attempts}.")

            ## block tokens are going to be the final new tokens we append to the current tokens
            block_tokens = None
            if use_rejection_sampling:
                infer_outputs = piref_model.generate(input_ids=[model_input], sampling_params=sampling_params)
                block_tokens = infer_outputs[0]["output_ids"]
                if not block_tokens:
                    break
                if len(block_tokens) > max_new_tokens:
                    block_tokens = block_tokens[:max_new_tokens]
                tmp_current_tokens = model_input.copy()
                tmp_current_tokens.extend(block_tokens)
                input_ids = torch.tensor(tmp_current_tokens, device=classifier_model.device, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones(input_ids.shape, device=classifier_model.device, dtype=torch.long)
                current_reward = classifier_model(input_ids=input_ids, attention_mask=attention_mask).success_probs[0, -1].item()
                ## here do rejection sampling if use_rejection_sampling is True
                while random.random() > (current_reward / max_reward):
                    print(f"Rejection sampling at attempt {attempts}.")
                    infer_outputs = piref_model.generate(input_ids=[model_input], sampling_params=sampling_params)
                    block_tokens = infer_outputs[0]["output_ids"]
                    if not block_tokens:
                        break
                    if len(block_tokens) > max_new_tokens:
                        block_tokens = block_tokens[:max_new_tokens]
                    tmp_input_ids = model_input.copy()
                    tmp_input_ids.extend(block_tokens)
                    input_ids = torch.tensor(tmp_input_ids, device=classifier_model.device, dtype=torch.long).unsqueeze(0)
                    attention_mask = torch.ones(input_ids.shape, device=classifier_model.device, dtype=torch.long)
                    current_reward = classifier_model(input_ids=input_ids, attention_mask=attention_mask).success_probs[0, -1].item()


            else:
                block_tokens = max_reward_block_tokens

            current_tokens.extend(block_tokens)
            block_history.append(block_tokens)
            blocks_kept += 1

            if current_tokens and current_tokens[-1] in stop_token_set:
                break

        outputs.append(current_tokens)
        ## returning the current tokens, i.e. a list of lists of ints for each prompt. Note that current tokens does not 
        ## include the prompt tokens.

        ## when you want to pass current_tokens to score_all_generations, you need to pass it as a list of lists of ints for each prompt.

    final_generated_ids = torch.tensor(outputs[0], device=classifier_model.device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(final_generated_ids.shape, device=classifier_model.device, dtype=torch.long)
    classifier_outputs = classifier_model(input_ids=final_generated_ids, attention_mask=attention_mask)
    scores = classifier_outputs.success_probs[0, :].tolist()
    print(f"Score2: {scores[-20:]}")
    print(f"shape of scores: {len(scores)}")
    print(f"length of final generated ids: {len(final_generated_ids[0])}")

    return outputs




@torch.no_grad()
def main(args: Args) -> None:
    # Set CUDA_VISIBLE_DEVICES before importing torch/sglang operations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = 'cuda:0'
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base_dataset = benchmark_data.get_dataset(args.benchmark)
    all_indices = list(range(len(base_dataset)))
    if args.max_samples > 0:
        print(f"Using first {args.max_samples} samples from the dataset.")
        all_indices = all_indices[:min(args.max_samples, len(all_indices))]
    args.dataset_size = len(all_indices)
        

    processed_indices = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if "data_idx" in record:
                    processed_indices.add(record["data_idx"])

    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
    if not remaining_indices:
        print("All samples already processed. Nothing to do.")
        print('before returning, logging to wandb...')
        log_to_server(args)
        print('after logging to wandb, returning...')
        return

    print(f"Processing {len(remaining_indices)} of {len(all_indices)} requested samples.")


    #print('before loading piref_model...')
    #torch.cuda.set_device(4) 
    #print('after setting device to 4')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.piref_model, padding_side="left")
    piref_model = sgl.Engine(
        model_path=args.piref_model,
        dtype=dtype,
        mem_fraction_static=args.piref_gpu_util,
        random_seed=args.seed,
        skip_tokenizer_init=True,
    )
    stop_token_ids = [get_eos_token_id(tokenizer)]

    dataset = IndexedDataset(base_dataset, remaining_indices)

    def preprocess(example):
        formatted_problem = deepseek_utils.format_roll_in(example["problem"])
        input_ids = tokenizer(formatted_problem, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": input_ids,
            "problem": example["problem"],
            "answer": example["answer"],
            "data_idx": example["data_idx"],
        }

    def simple_collate(batch):
        processed = [preprocess(item) for item in batch]
        return {
            "input_ids": [item["input_ids"] for item in processed],
            "problem": [item["problem"] for item in processed],
            "answer": [item["answer"] for item in processed],
            "data_idx": [item["data_idx"] for item in processed],
        }

    max_value_estimate_num_attempts = args.max_value_estimate_num_attempts
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate)
    import classifier_lib
    dtype = torch.bfloat16
    model_loading_kwargs = dict(attn_implementation=args.attention_impl, torch_dtype=dtype, use_cache=True)
    classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(args.classifier_ckpt_path, **model_loading_kwargs).to(device)
    with open(args.output_path, "a") as f_out:
        for batch in loader:
            batch_start = time.time()
            prompt_ids = batch["input_ids"]
            generated_ids = generate_single_line(
                piref_model=piref_model,
                prompt_ids=prompt_ids,
                max_length=args.max_length,
                block_size=args.block_size,
                max_blocks=args.max_blocks,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_ids=stop_token_ids,
                classifier_model=classifier,
                max_value_estimate_num_attempts=max_value_estimate_num_attempts,
                use_rejection_sampling=args.use_rejection_sampling,
            )
            elapsed = time.time() - batch_start
            per_sample_time = elapsed / max(len(prompt_ids), 1)

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            generated_solutions = [deepseek_utils.remove_thinking_text(text) for text in generated_texts]
            processed_answers = [accuracy_utils.process_sample(solution) for solution in generated_solutions]
            rewards = [
                accuracy_utils.math_verify_check(ground_truth, processed)
                for ground_truth, processed in zip(batch["answer"], processed_answers)
            ]
            generated_scores = score_all_generations(classifier, prompt_ids, [[ids] for ids in generated_ids], device, dtype)

            ## the only difference here with inference_eval.py is that generated ids for each prompt has only one generation, so that's
            ## why we need to pass the generated scores as a list of lists of ints for each prompt.

            for i in range(len(prompt_ids)):
                record = {
                    "data_idx": batch["data_idx"][i],
                    "problem": batch["problem"][i],
                    "gt_answer": batch["answer"][i],
                    "generated_ids": generated_ids[i],
                    "generated_scores": generated_scores[i],
                    "generated_raw_text": generated_texts[i],
                    "processed_answer": processed_answers[i],
                    "reward": rewards[i],
                    "dt": per_sample_time,
                }
                f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            min_idx = batch["data_idx"][0]
            max_idx = batch["data_idx"][-1]
            mean_reward = sum(rewards) / max(len(rewards), 1)
            print(
                f"Completed indices {min_idx}-{max_idx} "
                f"| batch_size={len(prompt_ids)} "
                f"| time={per_sample_time:.2f}s per sample "
                f"| reward_mean={mean_reward:.3f}"
            )

    print("Finished processing all requested samples.")
    log_to_server(args)

def get_logger(logger_kwargs):
    print("logger kwargs: ", logger_kwargs)
    if wandb is None:
        raise ImportError("wandb is not installed. Please install it with: pip install wandb")

    # Optional: Login programmatically if API key is set in code
    # This is NOT necessary if you've run 'wandb login' or set WANDB_API_KEY env var
    # wandb.login(key="your-api-key-here")

    tags = logger_kwargs.pop('tags')
    group_tag = tags.pop("group_tag")
    experiment = logger_kwargs.pop('experiment')
    experiment_name = logger_kwargs.pop('name')
    
    print("Starting to log to wandb...")
    run = wandb.init(
        project=experiment,
        name=experiment_name,
        # Don't use fixed id to avoid conflicts with deleted runs
        # id=experiment_name,  # commented out to avoid conflicts
        tags=[group_tag] + [f"{k}:{v}" for k, v in tags.items()],
        config=tags,
        resume="allow",  # allow resuming if run exists, or create new if deleted
    )
    return run

def get_tag(args, dvts_n=1):
    tag = f"sz_{args.block_size}_rep_{args.num_repetitions}_temp_{args.temperature}"
    if dvts_n == 1:
        return tag
    else:
        assert dvts_n > 1
        return f"{tag}_dvtsn_{dvts_n}"

def log_to_server(args: Args):
    data = eval_helpers.load_jsonl(args.output_path)
    print(f"length of data: {len(data)} length of dataset_size * num_repetitions: {args.dataset_size * args.num_repetitions}")
    #assert len(data) == (args.dataset_size * args.num_repetitions), f"Expected {args.dataset_size * args.num_repetitions} items in output, but got {len(data)}. Check if the output file is complete."
    tag = get_tag(args, dvts_n=1)
    tags = {
        #"search": str(args.search_type),
        "block_size": str(args.block_size),
        "seed": str(args.seed),
        "group_tag": str(tag),
        "dvts_n": str(1),
        "timestamp": str(time.time()),
    }
    experiment_name = f"{tag}_seed_{args.seed}"
    logger_kwargs = dict(
        experiment=args.wandb_project,
        name=experiment_name,
        tags=tags,
    )
    run = get_logger(logger_kwargs)

    scores = np.array([d['generated_scores'] for d in data])
    rewards = np.array([d['reward'] for d in data])
    processed_answers = np.array([d['processed_answer'] for d in data])
    
    # In backtrack version, rewards is 1D (one reward per sample) and scores is 2D (scores per token)
    # Use the final classifier score for each sample
    final_classifier_scores = np.array([scores[i][-1] if len(scores[i]) > 0 else 0.0 for i in range(len(scores))])
    
    # Since there's only one generation per sample, bon_rewards is just the rewards
    # But we can still use classifier_bon with n=1 for consistency
    bon_rewards = torch.tensor([
        eval_helpers.classifier_bon(rewards=[rewards[i]], classifier_values=[final_classifier_scores[i]], n=1)[0]
        for i in range(len(data))
    ]).float()
    
    # num_repetitions is outer loop
    bon_rewards = bon_rewards.view(args.num_repetitions, args.dataset_size).transpose(0, 1)
    
    for i in range(bon_rewards.shape[0]):
        print(f'shape[1] of bon_rewards: {bon_rewards.shape[1]}')
        for j in range(bon_rewards.shape[1]):
            print(f"Logging bon_rewards_{i}...")
            run.log({f"{args.benchmark}/bon_rewards_rep_{j}": bon_rewards[i, j].item()}, step=i)
    run.finish()



if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

