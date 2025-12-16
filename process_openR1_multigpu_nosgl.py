import os
import multiprocessing as mp
from typing import List, Tuple, Sequence
import random
import time
from dataclasses import dataclass

# Import basic modules that don't depend on CUDA
import tyro
import datasets
import transformers
from inference_eval_backtrack import Args, get_eos_token_id
import deepseek_utils
import accuracy_utils
import classifier_lib
# Note: torch is imported inside worker_process




def generate_single_line(
    piref_model,  # HuggingFace PreTrainedModel
    tokenizer,  # HuggingFace tokenizer
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
    """
    Generate text using a HuggingFace model instead of sglang engine.
    
    Args:
        piref_model: HuggingFace PreTrainedModel (e.g., AutoModelForCausalLM)
        tokenizer: HuggingFace tokenizer for encoding/decoding
        prompt_ids: Sequence of prompt token ID lists
        max_length: Maximum total length
        block_size: Size of each generation block
        max_blocks: Maximum number of blocks
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stop_token_ids: Token IDs that stop generation
        max_value_estimate_num_attempts: Number of attempts for value estimation
        classifier_model: Classifier model for reward calculation
        use_rejection_sampling: Whether to use rejection sampling
        
    Returns:
        List of generated token ID lists (without prompt tokens)
    """
    import torch
    
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

            model_input = prompt + current_tokens

            ## estimate the maximum of the rewards
            max_reward = 0
            average_reward = 0
            max_reward_block_tokens = None

            ## estimate the maximum of the rewards using rejection sampling, also the average reward
            for _ in range(max_value_estimate_num_attempts):
                tmp_input_ids = model_input.copy()
                
                # HuggingFace generate call
                input_tensor = torch.tensor([model_input], device=piref_model.device, dtype=torch.long)
                with torch.no_grad():
                    generated = piref_model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        eos_token_id=stop_token_ids[0] if stop_token_ids else None,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    )
                # Extract only the new tokens (remove input tokens)
                block_tokens = generated[0][len(model_input):].cpu().tolist()
                
                if not block_tokens:
                    break
                # Stop at first stop token
                for i, token in enumerate(block_tokens):
                    if token in stop_token_set:
                        block_tokens = block_tokens[:i+1]
                        break
                if len(block_tokens) > max_new_tokens:
                    block_tokens = block_tokens[:max_new_tokens]
                    
                tmp_input_ids.extend(block_tokens)
                input_ids = torch.tensor(tmp_input_ids, device=classifier_model.device, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones(input_ids.shape, device=classifier_model.device, dtype=torch.long)
                reward_sample = classifier_model(input_ids=input_ids, attention_mask=attention_mask).success_probs[0, -1].item()
                print(f"adding reward sample {reward_sample} to average reward")
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
                # HuggingFace generate call
                input_tensor = torch.tensor([model_input], device=piref_model.device, dtype=torch.long)
                with torch.no_grad():
                    generated = piref_model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        eos_token_id=stop_token_ids[0] if stop_token_ids else None,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    )
                # Extract only the new tokens
                block_tokens = generated[0][len(model_input):].cpu().tolist()
                
                if not block_tokens:
                    break
                # Stop at first stop token
                for i, token in enumerate(block_tokens):
                    if token in stop_token_set:
                        block_tokens = block_tokens[:i+1]
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
                    # HuggingFace generate call
                    input_tensor = torch.tensor([model_input], device=piref_model.device, dtype=torch.long)
                    with torch.no_grad():
                        generated = piref_model.generate(
                            input_tensor,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            eos_token_id=stop_token_ids[0] if stop_token_ids else None,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        )
                    # Extract only the new tokens
                    block_tokens = generated[0][len(model_input):].cpu().tolist()
                    
                    if not block_tokens:
                        break
                    # Stop at first stop token
                    for i, token in enumerate(block_tokens):
                        if token in stop_token_set:
                            block_tokens = block_tokens[:i+1]
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

    ## this line use the fact that I am assuing there is only one element in outputs!!!
    final_generated_ids = torch.tensor(outputs[0], device=classifier_model.device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(final_generated_ids.shape, device=classifier_model.device, dtype=torch.long)
    classifier_outputs = classifier_model(input_ids=final_generated_ids, attention_mask=attention_mask)
    scores = classifier_outputs.success_probs[0, :].tolist()
    print(f"Score2: {scores[-20:]}")
    print(f"shape of scores: {len(scores)}")
    print(f"length of final generated ids: {len(final_generated_ids[0])}")

    return outputs



def worker_process(
    gpu_id: int,
    args: Args,
    indices: List[int],
    dataset_path: str,
    output_data: dict,
    lock: mp.Lock,
) -> None:
    """
    Worker process that processes a subset of dataset indices on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use (0-7)
        args: Configuration arguments
        indices: List of dataset indices to process on this GPU
        dataset_path: Path to the dataset
        output_data: Shared dictionary to store results {idx: {problem, processed_answer, reward}}
        lock: Multiprocessing lock for thread-safe dictionary access
    """
    # Set environment variables for this process
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500 + gpu_id)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    # Import torch and transformers
    import torch
    import transformers
    
    # Set the device to the specific GPU ID
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"[GPU {gpu_id}] GPU {gpu_id} not available. Only {torch.cuda.device_count()} GPUs found.")
        torch.cuda.set_device(gpu_id)
    
    device = f'cuda:{gpu_id}'
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed + gpu_id)
    random.seed(args.seed + gpu_id)
    
    print(f"[GPU {gpu_id}] Starting worker process for {len(indices)} samples")
    print(f"[GPU {gpu_id}] Using device: {device}")
    print(f"[GPU {gpu_id}] torch.cuda.current_device() = {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

    piref_model = None
    classifier = None
    tokenizer = None
    
    try:
        # Load dataset
        dataset = datasets.load_from_disk(dataset_path)
        dataset = dataset.select(indices)
        
        print(f"[GPU {gpu_id}] Dataset loaded: {len(dataset)} examples")
        
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.piref_model)
        
        # Stagger initialization to avoid race conditions
        time.sleep(gpu_id * 0.5)
        
        # Initialize HuggingFace model on this GPU
        available_gb = None
        total_gb = None
        
        try:
            # Ensure CUDA context is initialized on the correct device
            if torch.cuda.is_available():
                # Set the device explicitly
                torch.cuda.set_device(gpu_id)
                # Create a dummy tensor on the target device to ensure CUDA context is initialized
                with torch.cuda.device(gpu_id):
                    dummy = torch.zeros(1, device=f'cuda:{gpu_id}')
                    del dummy
                    torch.cuda.synchronize()
                # Double-check that current device is correct
                current_device = torch.cuda.current_device()
                if current_device != gpu_id:
                    raise RuntimeError(f"[GPU {gpu_id}] Device mismatch: torch.cuda.current_device()={current_device}, expected={gpu_id}")
                print(f"[GPU {gpu_id}] CUDA context initialized on device {gpu_id} (verified: current_device={current_device})")
            
            # Check available memory before loading model
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            available_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            
            print(f"[GPU {gpu_id}] Memory before model load: Available={available_gb:.2f} GB, Total={total_gb:.2f} GB")
            
            # Load HuggingFace model
            # Note: generate_single_line expects piref_model.generate() method which is sglang-specific.
            # This HuggingFace model may not work directly with generate_single_line without modifications.
            print(f"[GPU {gpu_id}] Loading HuggingFace model from {args.piref_model}...")
            piref_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.piref_model,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            print(f"[GPU {gpu_id}] HuggingFace model loaded successfully")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[GPU {gpu_id}] Error: Out of memory when loading model")
                if available_gb is not None and total_gb is not None:
                    print(f"[GPU {gpu_id}] Available memory: {available_gb:.2f} GB, Total: {total_gb:.2f} GB")
            raise
        except Exception as e:
            print(f"[GPU {gpu_id}] Error loading HuggingFace model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Load classifier model
        model_loading_kwargs = dict(attn_implementation=args.attention_impl, torch_dtype=dtype, use_cache=True)
        classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(
            args.classifier_ckpt_path, **model_loading_kwargs
        ).to(device)
        print(f"[GPU {gpu_id}] Classifier model loaded")
        
        # Process each example
        for idx, example in enumerate(dataset):
            original_idx = indices[idx]
            problem = example["problem"]
            ground_truth = example.get("answer", "")
            
            try:
                # Format the problem
                formatted_problem = deepseek_utils.format_roll_in(problem)
                input_ids = tokenizer(formatted_problem, add_special_tokens=False)["input_ids"]
                
                print(f"[GPU {gpu_id}] Processing example {original_idx} ({idx+1}/{len(dataset)})")
                
                # Generate solution using HuggingFace model
                
                #generated_ids = generate_single_line(
                #    piref_model=piref_model,
                #    tokenizer=tokenizer,
                #    prompt_ids=[input_ids],
                #    max_length=args.max_length,
                #    block_size=args.block_size,
                #    max_blocks=args.max_blocks,
                #    temperature=args.temperature,
                #    top_p=args.top_p,
                #    stop_token_ids=[tokenizer.eos_token_id],
                #    max_value_estimate_num_attempts=args.max_value_estimate_num_attempts,
                #    classifier_model=classifier,
                #    use_rejection_sampling=args.use_rejection_sampling,
                #)

                ## uncomment this line if you want to skip the prm generation and just sample from the base model
                generated_ids = piref_model.generate(input_ids=torch.tensor([input_ids]).to(device), max_new_tokens=args.max_length, temperature=args.temperature, top_p=args.top_p, do_sample=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id).tolist()


                # Decode and process the generated solution
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
                generated_solution = deepseek_utils.remove_thinking_text(generated_text)
                processed_answer = accuracy_utils.process_sample(generated_solution)
                
                # Calculate reward (if ground truth available)
                if ground_truth:
                    reward = accuracy_utils.math_verify_check(ground_truth, generated_solution)
                else:
                    reward = 0.0
                
                # Store result
                with lock:
                    output_data[original_idx] = {
                        "problem": problem,
                        "processed_answer": processed_answer,
                        "reward": reward,
                        "generated_solution": generated_solution,
                    }
                
                print(f"[GPU {gpu_id}] Completed example {original_idx} | reward={reward:.3f}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing example {original_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Store error result
                with lock:
                    output_data[original_idx] = {
                        "problem": problem,
                        "processed_answer": "",
                        "reward": 0.0,
                        "generated_solution": "",
                    }
                continue
        
        print(f"[GPU {gpu_id}] Finished processing all assigned samples")
        
    except KeyboardInterrupt:
        print(f"[GPU {gpu_id}] Worker process interrupted by user")
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error in worker process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up models and resources - this always runs
        print(f"[GPU {gpu_id}] Starting cleanup...")
        
        try:
            # Clean up HuggingFace model
            if piref_model is not None:
                del piref_model
                print(f"[GPU {gpu_id}] HuggingFace model deleted")
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Error cleaning up model: {e}")

        # Clean up GPU memory
        try:
            if classifier is not None:
                del classifier
            if tokenizer is not None:
                del tokenizer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Error during cleanup: {e}")
        
        print(f"[GPU {gpu_id}] Worker process exiting")
    return  # Explicit return to signal function completion

def check_gpu_available(gpu_id: int, args: Args) -> Tuple[bool, float, float]:
    """
    Check if a GPU has sufficient memory for the given args.
    
    Returns:
        Tuple of (has_sufficient_memory, available_memory_gb, required_memory_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0.0, 0.0
        if gpu_id >= torch.cuda.device_count():
            return False, 0.0, 0.0
        
        # Use the GPU ID directly
        free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
        available_gb = free_memory / (1024**3)
        total_gb = total_memory / (1024**3)
        
        # Relaxed memory requirements for HuggingFace model
        # Reduced estimates to allow more GPUs to be used
        model_base_memory_gb = 5.0  # Reduced from 8.0 - base model memory (approximate for 1.5B model)
        inference_overhead_gb = 2.0  # Reduced from 4.0 - Memory for inference (activations, KV cache, etc.)
        
        # Classifier memory (more conservative estimates)
        classifier_base_memory_gb = 2.5  # Reduced from 3.5
        max_seq_length = args.max_length
        classifier_kv_cache_per_pass_gb = (max_seq_length / 1000) * 0.3  # Reduced from 0.4
        classifier_activation_per_pass_gb = (max_seq_length / 1000) * 0.15  # Reduced from 0.2
        max_concurrent_passes = max(args.max_value_estimate_num_attempts, 2)
        if args.use_rejection_sampling:
            max_concurrent_passes = max(max_concurrent_passes, 3)
        classifier_peak_memory_gb = classifier_base_memory_gb + (classifier_kv_cache_per_pass_gb + classifier_activation_per_pass_gb) * max_concurrent_passes
        
        # Reduced buffer for more lenient checking
        buffer_gb = 2.0  # Reduced from 8.0
        required_gb = model_base_memory_gb + inference_overhead_gb + classifier_peak_memory_gb + buffer_gb
        
        # Very lenient check: accept GPUs with at least 60% of calculated requirement, or minimum 8 GB available
        # This allows GPUs with less headroom to still be used
        minimum_required_gb = min(required_gb * 0.6, required_gb - 6.0)  # Allow 40% reduction or 6GB less
        minimum_available_gb = 8.0  # Minimum 8 GB free memory required
        
        # Check if there's sufficient memory (very lenient)
        if available_gb < minimum_available_gb:
            has_sufficient = False
        elif available_gb >= required_gb:
            has_sufficient = True
        elif available_gb >= minimum_required_gb:
            # Accept GPU if it has at least 60% of required memory
            has_sufficient = True
        else:
            # Even more lenient: accept if available memory is at least 6 GB
            # This is a very permissive check to maximize GPU utilization
            has_sufficient = available_gb >= 6.0
        
        return has_sufficient, available_gb, required_gb
    except Exception as e:
        print(f"Warning: Could not check GPU {gpu_id} memory: {e}")
        return False, 0.0, 0.0


def main_multigpu(args: Args, num_samples: int = 100, gpu_list: List[int] = None) -> None:
    """
    Main function that processes dataset using multiple GPUs.
    
    Args:
        args: Configuration arguments
        num_samples: Number of samples to process (default: 100)
        gpu_list: Optional list of specific GPU IDs to use
    """
    # Use 'spawn' method to ensure each process starts fresh
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    dataset_path = '/data/datasets/openR1_math_220k'
    output_path = 'processed_dataset_multigpu_nosgl_withprm.jsonl'
    
    # Load dataset to get indices
    print(f"Loading dataset from {dataset_path}...")
    dataset = datasets.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(dataset)} total examples")
    
    # Select first num_samples
    num_samples = min(num_samples, len(dataset))
    indices_to_process = list(range(num_samples))
    print(f"Processing first {num_samples} examples")
    
    # Determine which GPUs to use
    import torch
    if gpu_list is None:
        # Auto-detect available GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        available_gpus = []
        for gpu_id in range(num_gpus):
            has_sufficient, available_gb, required_gb = check_gpu_available(gpu_id, args)
            if has_sufficient:
                available_gpus.append(gpu_id)
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✓ Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Margin={margin:.2f} GB")
            else:
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✗ Insufficient memory - Available={available_gb:.2f} GB, Required={required_gb:.2f} GB")
        gpu_list = available_gpus if available_gpus else []
        if not gpu_list:
            raise RuntimeError("No GPUs have sufficient memory. Please free up GPU memory or use fewer GPUs.")
        print(f"Using GPUs: {gpu_list}")
    else:
        # Validate specified GPUs
        print(f"Validating specified GPUs: {gpu_list}")
        valid_gpus = []
        for gpu_id in gpu_list:
            has_sufficient, available_gb, required_gb = check_gpu_available(gpu_id, args)
            if has_sufficient:
                valid_gpus.append(gpu_id)
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✓ Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Margin={margin:.2f} GB")
            else:
                print(f"GPU {gpu_id}: ✗ Insufficient memory")
        if not valid_gpus:
            raise RuntimeError("None of the specified GPUs have sufficient memory.")
        gpu_list = valid_gpus
    
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        raise ValueError("No available GPUs to use!")
    
    print(f"Starting multi-GPU processing on {num_gpus} GPUs")
    
    # Split indices across GPUs
    indices_per_gpu = len(indices_to_process) // num_gpus
    remainder = len(indices_to_process) % num_gpus
    
    gpu_indices = []
    start_idx = 0
    for idx, actual_gpu_id in enumerate(gpu_list):
        size = indices_per_gpu + (1 if idx < remainder else 0)
        end_idx = start_idx + size
        gpu_indices.append((actual_gpu_id, indices_to_process[start_idx:end_idx]))
        start_idx = end_idx
        print(f"GPU {actual_gpu_id}: {len(gpu_indices[-1][1])} samples")
    
    # Create shared dictionary and lock for results
    manager = mp.Manager()
    output_data = manager.dict()
    lock = manager.Lock()
    
    # Create and start processes
    processes = []
    for actual_gpu_id, indices in gpu_indices:
        if len(indices) == 0:
            print(f"Skipping GPU {actual_gpu_id} (no samples assigned)")
            continue
        
        p = mp.Process(
            target=worker_process,
            args=(actual_gpu_id, args, indices, dataset_path, output_data, lock),
        )
        p.start()
        processes.append((actual_gpu_id, p))
        print(f"Started process for GPU {actual_gpu_id}")
        time.sleep(0.1)
    
    print(f"Number of processes: {len(processes)}")
    
    if len(processes) == 0:
        print("ERROR: No processes were started! Check GPU selection logic.")
        return
    
    # Wait for all processes to complete with timeout
    timeout_seconds = 3600 * 2  # 2 hours timeout per process
    for gpu_id, p in processes:
        print(f"Joining process for GPU {gpu_id} (timeout: {timeout_seconds}s)")
        p.join(timeout=timeout_seconds)
        
        if p.is_alive():
            print(f"ERROR: GPU {gpu_id} process is still alive after timeout! Terminating...")
            p.terminate()
            p.join(timeout=5)  # Give it 5 seconds to terminate
            if p.is_alive():
                print(f"ERROR: GPU {gpu_id} process did not terminate, killing...")
                p.kill()
                p.join(timeout=5)
        
        if p.exitcode is None:
            print(f"WARNING: GPU {gpu_id} process exitcode is None (may have been killed)")
        elif p.exitcode != 0:
            print(f"WARNING: GPU {gpu_id} process exited with code {p.exitcode}")
        else:
            print(f"GPU {gpu_id} process completed successfully")
    
    print("All GPU processes finished.")
    
    # Collect results in order
    print(f"Collecting results from {len(output_data)} processed examples...")
    results = []
    for idx in range(num_samples):
        if idx in output_data:
            results.append(output_data[idx])
        else:
            print(f"Warning: Missing result for index {idx}")
    
    # Create new dataset from results
    print(f"Creating new dataset with {len(results)} examples...")
    
    # Create a new dataset with the processed data
    new_dataset_dict = {
        "problem": [r["problem"] for r in results],
        "processed_answer": [r["processed_answer"] for r in results],
        "reward": [r["reward"] for r in results],
        #"generated_solution": [r["generated_solution"] for r in results],
    }
    
    # Explicitly define schema to ensure reward is float32, not boolean
    from datasets import Features, Value
    features = Features({
        "problem": Value("string"),
        "processed_answer": Value("string"),
        "reward": Value("float32"),
    })
    
    new_dataset = datasets.Dataset.from_dict(new_dataset_dict, features=features)
    
    print(f"\nNew dataset created: {len(new_dataset)} examples")
    print(f"Column names: {new_dataset.column_names}")
    if len(new_dataset) > 0:
        print(f"\nFirst example:\n{new_dataset[0]}")
    
    # Save dataset
    avoid_prm = True
    if avoid_prm:
        output_dataset_path = "processed_openR1_100_multigpu_nosgl_avoid_prm"
    else:
        output_dataset_path = "processed_openR1_100_multigpu_nosgl"
    new_dataset.to_json(output_path)
    new_dataset.save_to_disk(output_dataset_path)
    print(f"\nDataset saved to {output_path} and {output_dataset_path}")
    print("Processing complete! Avoiding PRM")


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Auto-detect available GPUs, or specify a list like [0, 1, 2, 3]
    main_multigpu(args, num_samples=5, gpu_list=None)

