import os
import multiprocessing as mp
from typing import List, Tuple
import random
import time
from dataclasses import dataclass

# Import basic modules that don't depend on CUDA
import tyro
import ujson as json

# Import functions from the original file
# Note: The original file sets CUDA_VISIBLE_DEVICES="4" at import time, but this only
# affects the main process (which doesn't use GPU). Each worker process will set
# CUDA_VISIBLE_DEVICES before importing torch/sglang.
from inference_eval_backtrack import (
    Args,
    get_eos_token_id,
    IndexedDataset,
    score_all_generations,
    generate_single_line,
    log_to_server,
)

import benchmark_data
import deepseek_utils
import accuracy_utils


def worker_process(
    gpu_id: int,
    args: Args,
    indices: List[int],
    output_path: str,
    lock: mp.Lock,
) -> None:
    """
    Worker process that processes a subset of indices on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use (0-7)
        args: Configuration arguments
        indices: List of dataset indices to process on this GPU
        output_path: Path to output file (shared across processes)
        lock: Multiprocessing lock for file writing
    """
    # Set CUDA device for this process BEFORE importing torch/sglang
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Disable NCCL/PyTorch distributed to avoid conflicts in multi-process setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500 + gpu_id)  # Different port per GPU
    # Try to disable distributed mode in sglang if possible
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    
    # Now import torch and sglang after setting CUDA_VISIBLE_DEVICES
    import torch
    import transformers
    import sglang as sgl
    from torch.utils.data import DataLoader
    import classifier_lib
    
    # Explicitly set the device after torch import
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use device 0 since CUDA_VISIBLE_DEVICES restricts visibility
    
    device = f'cuda:0'  # Always use cuda:0 since CUDA_VISIBLE_DEVICES restricts visibility
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    torch.set_float32_matmul_precision("high")
    # Use different seed per GPU to ensure diversity
    torch.manual_seed(args.seed + gpu_id)
    random.seed(args.seed + gpu_id)
    
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, torch.cuda.device_count()={torch.cuda.device_count()}")

    print(f"[GPU {gpu_id}] Processing {len(indices)} samples: {indices[:5]}...")

    # Load dataset
    base_dataset = benchmark_data.get_dataset(args.benchmark)
    
    # Check which indices are already processed (with lock to avoid race conditions)
    processed_indices = set()
    if os.path.exists(output_path):
        with lock:
            try:
                with open(output_path, "r") as f:
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
            except Exception as e:
                print(f"[GPU {gpu_id}] Error reading output file: {e}")

    # Filter out already processed indices
    remaining_indices = [idx for idx in indices if idx not in processed_indices]
    if not remaining_indices:
        print(f"[GPU {gpu_id}] All assigned samples already processed. Exiting.")
        return

    print(f"[GPU {gpu_id}] Processing {len(remaining_indices)} of {len(indices)} assigned samples.")

    # Load tokenizer and models
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.piref_model, padding_side="left")
    
    # Initialize sglang engine on this GPU
    # Add a small delay to avoid race conditions when multiple processes start simultaneously
    time.sleep(gpu_id * 0.5)  # Stagger initialization by 0.5s per GPU
    
    # Note: sglang.Engine should respect CUDA_VISIBLE_DEVICES
    # If it doesn't, we may need to pass device explicitly or use a different approach
    try:
        piref_model = sgl.Engine(
            model_path=args.piref_model,
            dtype=dtype,
            mem_fraction_static=args.piref_gpu_util,
            random_seed=args.seed + gpu_id,
            skip_tokenizer_init=True,
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] Error initializing sglang engine: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to fail fast and see the actual error
    
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
    
    # Load classifier model
    model_loading_kwargs = dict(attn_implementation=args.attention_impl, torch_dtype=dtype, use_cache=True)
    classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(
        args.classifier_ckpt_path, **model_loading_kwargs
    ).to(device)
    
    # Process batches
    for batch in loader:
        batch_start = time.time()
        prompt_ids = batch["input_ids"]
        
        try:
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
        except Exception as e:
            print(f"[GPU {gpu_id}] Error in generate_single_line: {e}")
            continue
        
        elapsed = time.time() - batch_start
        per_sample_time = elapsed / max(len(prompt_ids), 1)

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        generated_solutions = [deepseek_utils.remove_thinking_text(text) for text in generated_texts]
        processed_answers = [accuracy_utils.process_sample(solution) for solution in generated_solutions]
        rewards = [
            accuracy_utils.math_verify_check(ground_truth, processed)
            for ground_truth, processed in zip(batch["answer"], processed_answers)
        ]
        generated_scores = score_all_generations(
            classifier, prompt_ids, [[ids] for ids in generated_ids], device, dtype
        )

        # Write results with lock to avoid conflicts
        with lock:
            try:
                with open(output_path, "a") as f_out:
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
            except Exception as e:
                print(f"[GPU {gpu_id}] Error writing to output file: {e}")

        min_idx = batch["data_idx"][0]
        max_idx = batch["data_idx"][-1]
        mean_reward = sum(rewards) / max(len(rewards), 1)
        print(
            f"[GPU {gpu_id}] Completed indices {min_idx}-{max_idx} "
            f"| batch_size={len(prompt_ids)} "
            f"| time={per_sample_time:.2f}s per sample "
            f"| reward_mean={mean_reward:.3f}"
        )

    print(f"[GPU {gpu_id}] Finished processing all assigned samples.")


def check_gpu_has_sufficient_memory(gpu_id: int, args: Args) -> Tuple[bool, float, float]:
    """
    Check if a GPU has sufficient memory for the given args.
    Same logic as in inference_eval_backtrack_backup.py
    
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
            buffer_gb = 8.0  # Increased buffer to prevent runtime OOM
            
            # Required memory: sglang reservation + classifier peak + buffer
            required_gb = sglang_reservation_gb + classifier_peak_memory_gb + buffer_gb
            
            # Additional check: if sglang's reservation alone is close to available, it's risky
            if sglang_reservation_gb > available_gb * 0.85:  # More conservative: 85% threshold
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


def check_gpu_available(gpu_id: int, min_free_memory_gb: float = 5.0) -> bool:
    """Check if a GPU is available with sufficient free memory (legacy function)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if gpu_id >= torch.cuda.device_count():
            return False
        # Temporarily set CUDA_VISIBLE_DEVICES to check this specific GPU
        old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(0)  # Use 0 since CUDA_VISIBLE_DEVICES restricts
            free_gb = free_memory / (1024**3)
            return free_gb >= min_free_memory_gb
        finally:
            if old_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
    except Exception as e:
        print(f"Warning: Could not check GPU {gpu_id} availability: {e}")
        return False


def main_multigpu(args: Args, num_gpus: int = 8, gpu_list: List[int] = None) -> None:
    """
    Main function that distributes work across multiple GPUs.
    
    Args:
        args: Configuration arguments
        num_gpus: Number of GPUs to use (default: 8)
        gpu_list: Optional list of specific GPU IDs to use (e.g., [1,2,3,5,6,7])
                  If None, uses GPUs 0 through num_gpus-1
    """
    # Use 'spawn' method to ensure each process starts fresh (important for CUDA)
    # This ensures CUDA_VISIBLE_DEVICES is set before torch/sglang imports in each worker
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Determine which GPUs to use
    if gpu_list is None:
        # Check available GPUs with comprehensive memory check
        import torch
        available_gpus = []
        for gpu_id in range(num_gpus):
            has_sufficient, available_gb, required_gb = check_gpu_has_sufficient_memory(gpu_id, args)
            if has_sufficient:
                available_gpus.append(gpu_id)
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✓ Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Margin={margin:.2f} GB")
            else:
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✗ Insufficient memory - Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Shortage={abs(margin):.2f} GB")
        gpu_list = available_gpus if available_gpus else []
        if not gpu_list:
            raise RuntimeError("No GPUs have sufficient memory for the given args. Please reduce piref_gpu_util, max_length, or use GPUs with more available memory.")
        print(f"Using GPUs with sufficient memory: {gpu_list}")
    else:
        # Validate specified GPUs have sufficient memory
        print(f"Validating specified GPUs: {gpu_list}")
        valid_gpus = []
        for gpu_id in gpu_list:
            has_sufficient, available_gb, required_gb = check_gpu_has_sufficient_memory(gpu_id, args)
            if has_sufficient:
                valid_gpus.append(gpu_id)
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✓ Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Margin={margin:.2f} GB")
            else:
                margin = available_gb - required_gb
                print(f"GPU {gpu_id}: ✗ Insufficient memory - Available={available_gb:.2f} GB, Required={required_gb:.2f} GB, Shortage={abs(margin):.2f} GB")
                print(f"  Consider reducing piref_gpu_util (currently {args.piref_gpu_util}) or max_length (currently {args.max_length})")
        if not valid_gpus:
            raise RuntimeError("None of the specified GPUs have sufficient memory. Please adjust args or use different GPUs.")
        gpu_list = valid_gpus
        print(f"Using validated GPUs: {gpu_list}")
    
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        raise ValueError("No available GPUs to use!")
    
    print(f"Starting multi-GPU inference on {num_gpus} GPUs")
    
    # Load dataset and determine indices to process
    base_dataset = benchmark_data.get_dataset(args.benchmark)
    all_indices = list(range(len(base_dataset)))
    if args.max_samples > 0:
        print(f"Using first {args.max_samples} samples from the dataset.")
        all_indices = all_indices[:min(args.max_samples, len(all_indices))]
    args.dataset_size = len(all_indices)

    # Check already processed indices
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

    print(f"Processing {len(remaining_indices)} of {len(all_indices)} requested samples across {num_gpus} GPUs.")

    # Split indices across GPUs
    indices_per_gpu = len(remaining_indices) // num_gpus
    remainder = len(remaining_indices) % num_gpus
    
    gpu_indices = []
    start_idx = 0
    for idx, actual_gpu_id in enumerate(gpu_list):
        # Distribute remainder across first few GPUs
        size = indices_per_gpu + (1 if idx < remainder else 0)
        end_idx = start_idx + size
        gpu_indices.append((actual_gpu_id, remaining_indices[start_idx:end_idx]))
        start_idx = end_idx
        print(f"GPU {actual_gpu_id}: {len(gpu_indices[-1][1])} samples")

    # Create multiprocessing lock for file writing
    lock = mp.Lock()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and start processes
    processes = []
    for actual_gpu_id, indices in gpu_indices:
        if len(indices) == 0:
            print(f"Skipping GPU {actual_gpu_id} (no samples assigned)")
            continue
        
        p = mp.Process(
            target=worker_process,
            args=(actual_gpu_id, args, indices, args.output_path, lock),
        )
        p.start()
        processes.append((actual_gpu_id, p))
        print(f"Started process for GPU {actual_gpu_id}")
        # Small delay between process starts to avoid initialization conflicts
        time.sleep(0.1)

    # Wait for all processes to complete
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: GPU {gpu_id} process exited with code {p.exitcode}")
        else:
            print(f"GPU {gpu_id} process completed successfully")

    print("All GPU processes finished.")
    print("Finished processing all requested samples.")
    
    # Only log to server if output file exists and has data
    if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
        log_to_server(args)
    else:
        print(f"Warning: Output file {args.output_path} does not exist or is empty. Skipping wandb logging.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Use GPUs 1,2,3,5,6,7 to avoid GPUs 0 and 4 which are heavily used
    # You can modify this list or set to None to auto-detect available GPUs
    main_multigpu(args, num_gpus=8, gpu_list=[1, 2, 3, 5, 6, 7])  # Skip GPUs 0 and 4

