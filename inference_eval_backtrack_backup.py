import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple
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
    max_samples: int = 2
    seed: int = 1337
    piref_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    output_path: str = "outputs/inference_outputs.jsonl"
    attention_impl: str = "sdpa"
    piref_gpu_util: float = 0.5
    batch_size: int = 1
    max_length: int = 4096 ##16384
    block_size: int = 1024 ##4096
    max_blocks: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    classifier_ckpt_path: str = "VGS-AI/DeepSeek-VM-1.5B"
    use_rejection_sampling: bool = False
    max_value_estimate_num_attempts: int = 1
    num_repetitions: int = 1 
    dataset_size: int = -1
    wandb_project: str = "PRM_prediction_AME24_backtrack"
    gpu_id: int = 4

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


def estimate_required_memory_gb(args: Args, total_gpu_memory_gb: float) -> float:
    """
    Estimate required GPU memory in GB based on args parameters.
    
    This estimates memory for:
    - piref_model (sglang engine): uses piref_gpu_util fraction of GPU memory
      (this already includes KV cache and internal buffers)
    - classifier_model: model weights + KV cache + activations
    - Peak memory during rejection sampling (multiple forward passes)
    - Buffer for safety
    
    Args:
        args: Configuration arguments
        total_gpu_memory_gb: Total GPU memory in GB
        
    Returns:
        Estimated required memory in GB
    """
    # piref_model memory: uses piref_gpu_util fraction of total GPU memory
    # Note: sglang's mem_fraction_static already accounts for KV cache and buffers
    piref_memory_gb = total_gpu_memory_gb * args.piref_gpu_util
    
    # classifier_model memory: estimate based on model size
    # 1.5B model in bfloat16 ≈ 3GB base for weights
    classifier_base_memory_gb = 3.5  # Model weights + small overhead
    
    # KV cache for classifier during forward pass
    # For transformer models: 2 * num_layers * hidden_size * seq_len * 2 bytes (bfloat16)
    # For 1.5B model: ~24 layers, 1536 hidden_size
    # Rough estimate: ~0.4GB per 1000 tokens for KV cache
    max_seq_length = args.max_length
    classifier_kv_cache_per_pass_gb = (max_seq_length / 1000) * 0.4
    
    # Activation memory during forward pass (intermediate activations)
    # Rough estimate: ~0.2GB per 1000 tokens for activations
    classifier_activation_per_pass_gb = (max_seq_length / 1000) * 0.2
    
    # Peak memory during rejection sampling: multiple forward passes
    # Account for max_value_estimate_num_attempts + potential rejection sampling loops
    # In worst case, we might have multiple classifier forward passes
    max_concurrent_passes = max(args.max_value_estimate_num_attempts, 2)
    if args.use_rejection_sampling:
        max_concurrent_passes = max(max_concurrent_passes, 3)  # Rejection sampling can loop
    
    classifier_peak_memory_gb = classifier_base_memory_gb + (classifier_kv_cache_per_pass_gb + classifier_activation_per_pass_gb) * max_concurrent_passes
    
    # Safety buffer for PyTorch memory allocator fragmentation and other overhead
    # Increased buffer to account for memory fragmentation
    buffer_gb = 4.0
    
    total_required_gb = piref_memory_gb + classifier_peak_memory_gb + buffer_gb
    
    return total_required_gb


def check_gpu_has_sufficient_memory(gpu_id: int, args: Args) -> Tuple[bool, float, float]:
    """
    Check if a GPU has sufficient memory for the given args.
    
    Args:
        gpu_id: GPU ID to check
        args: Configuration arguments
        
    Returns:
        Tuple of (has_sufficient_memory, available_memory_gb, required_memory_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0.0, 0.0
        if gpu_id >= torch.cuda.device_count():
            return False, 0.0, 0.0
        
        # Temporarily set CUDA_VISIBLE_DEVICES to check this specific GPU
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # Get available memory on this GPU
            free_memory, total_memory = torch.cuda.mem_get_info(0)  # Use 0 since CUDA_VISIBLE_DEVICES restricts
            available_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            
            # sglang will try to reserve this much memory (based on total, not available)
            sglang_reservation_gb = total_gb * args.piref_gpu_util
            
            # Estimate classifier memory needs
            classifier_base_memory_gb = 3.5
            max_seq_length = args.max_length
            classifier_kv_cache_per_pass_gb = (max_seq_length / 1000) * 0.4
            classifier_activation_per_pass_gb = (max_seq_length / 1000) * 0.2
            max_concurrent_passes = max(args.max_value_estimate_num_attempts, 2)
            if args.use_rejection_sampling:
                max_concurrent_passes = max(max_concurrent_passes, 3)
            classifier_peak_memory_gb = classifier_base_memory_gb + (classifier_kv_cache_per_pass_gb + classifier_activation_per_pass_gb) * max_concurrent_passes
            
            # Safety buffer for memory fragmentation, peak usage, and temporary allocations
            # Increased buffer to account for:
            # - Memory fragmentation (can reduce effective memory by 10-20%)
            # - Temporary tensors during generation
            # - Multiple forward passes during rejection sampling
            # - PyTorch memory allocator overhead
            # - Memory spikes during long sequences
            buffer_gb = 8.0  # Increased buffer to prevent runtime OOM
            
            # Required memory calculation:
            # sglang and classifier share the GPU, but sglang's reservation is a hard limit
            # We need space for both models plus buffer
            required_gb = sglang_reservation_gb + classifier_peak_memory_gb + buffer_gb
            
            # Additional check: if sglang's reservation alone is close to available, it's risky
            # sglang might not be able to allocate its full reservation if memory is fragmented
            # Also check if required exceeds available
            if sglang_reservation_gb > available_gb * 0.85:  # More conservative: 85% threshold
                # sglang wants >85% of available - very risky, likely to fail
                # Mark as insufficient
                has_sufficient = False
            elif required_gb > available_gb:
                has_sufficient = False
            else:
                has_sufficient = available_gb >= required_gb
            
            return has_sufficient, available_gb, required_gb
        finally:
            if old_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
    except Exception as e:
        print(f"Warning: Could not check GPU {gpu_id} memory: {e}")
        return False, 0.0, 0.0


@torch.no_grad()
def main(args: Args) -> None:

    import os
    # Check GPU memory before setting CUDA_VISIBLE_DEVICES
    has_sufficient, available_gb, required_gb = check_gpu_has_sufficient_memory(args.gpu_id, args)
    
    # Add extra safety margin (10% of required memory) to account for memory fragmentation
    safety_margin_gb = required_gb * 0.1
    total_required_with_margin = required_gb + safety_margin_gb
    
    if available_gb < total_required_with_margin:
        print(f"ERROR: GPU {args.gpu_id} does not have sufficient memory!")
        print(f"  Required (base): {required_gb:.2f} GB")
        print(f"  Required (with safety margin): {total_required_with_margin:.2f} GB")
        print(f"  Available: {available_gb:.2f} GB")
        print(f"  Shortage: {total_required_with_margin - available_gb:.2f} GB")
        print("\nMemory breakdown:")
        print(f"  - piref_model (sglang): ~{required_gb * 0.4:.2f} GB")
        print(f"  - classifier_model (peak): ~{required_gb * 0.3:.2f} GB")
        print(f"  - KV cache & activations: ~{required_gb * 0.2:.2f} GB")
        print(f"  - Safety buffer: ~{required_gb * 0.1:.2f} GB")
        print("\nPlease either:")
        print("  1. Use a GPU with more available memory (set --gpu_id to a different GPU)")
        print("  2. Reduce piref_gpu_util (currently {})".format(args.piref_gpu_util))
        print("  3. Reduce max_length (currently {})".format(args.max_length))
        print("  4. Reduce batch_size (currently {})".format(args.batch_size))
        print("  5. Reduce max_value_estimate_num_attempts (currently {})".format(args.max_value_estimate_num_attempts))
        raise RuntimeError(f"GPU {args.gpu_id} has insufficient memory: {available_gb:.2f} GB available, {total_required_with_margin:.2f} GB required (with margin)")
    
    margin_gb = available_gb - total_required_with_margin
    if margin_gb < 5.0:
        print(f"WARNING: GPU {args.gpu_id} has low memory margin!")
        print(f"  Required (with margin): {total_required_with_margin:.2f} GB")
        print(f"  Available: {available_gb:.2f} GB")
        print(f"  Margin: {margin_gb:.2f} GB (recommended: >5 GB)")
        print("  Consider reducing memory usage to avoid potential OOM errors.")
    
    print(f"GPU {args.gpu_id} memory check passed:")
    print(f"  Required (base): {required_gb:.2f} GB")
    print(f"  Required (with margin): {total_required_with_margin:.2f} GB")
    print(f"  Available: {available_gb:.2f} GB")
    print(f"  Margin: {margin_gb:.2f} GB")
    
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
    num_beams = rewards.shape[1]
    bon_rewards = torch.tensor([
        eval_helpers.classifier_bon(rewards=rewards[i], classifier_values=scores[i], n=num_beams)
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

