import os
import multiprocessing as mp
from typing import List, Tuple
import random
import time
from dataclasses import dataclass

# Import basic modules that don't depend on CUDA
import tyro
import datasets
import transformers
from inference_eval_backtrack import Args, get_eos_token_id, generate_single_line
import deepseek_utils
import accuracy_utils
import classifier_lib
# Note: torch and sglang are imported inside worker_process


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
    # Disable sglang's multi-GPU memory imbalance check (we're running separate processes per GPU)
    os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"
    
    # Import torch and sglang
    import torch
    import transformers
    import sglang as sgl
    
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
        
        # Load tokenizer and models
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.piref_model)
        
        # Stagger initialization to avoid race conditions
        time.sleep(gpu_id * 0.5)
        
        # Initialize sglang engine on this GPU
        # Note: sglang.Engine may have issues with device parameter in some versions
        # We set torch.cuda.set_device(gpu_id) above, so sglang should use the correct device
        # Ensure device is set and create a dummy tensor to ensure CUDA context is initialized
        available_gb = None
        total_gb = None
        effective_fraction = args.piref_gpu_util
        
        try:
            # Ensure CUDA context is initialized on the correct device
            # This is critical for sglang to detect the correct GPU
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
            
            # Check available memory before initializing sglang
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            available_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            requested_fraction = args.piref_gpu_util
            requested_gb = total_gb * requested_fraction
            
            print(f"[GPU {gpu_id}] Memory before sglang init: Available={available_gb:.2f} GB, Total={total_gb:.2f} GB, Requested={requested_gb:.2f} GB ({requested_fraction*100:.1f}%)")
            
            # Initialize sglang without device parameter (it will use torch.cuda.current_device())
            # We've already set torch.cuda.set_device(gpu_id) above, so sglang should detect the correct GPU
            # The SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK env var disables multi-GPU memory balance checks
            
            # Initialize sglang engine
            # The bug in sglang's get_available_gpu_memory is an UnboundLocalError that happens
            # during initialization. We need to use sgl.Engine because generate_single_line requires it.
            # Ensure CUDA context is fully initialized
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    _ = torch.zeros(1, device=f'cuda:{gpu_id}')
                    torch.cuda.synchronize()
            
            # Initialize sglang - it will use torch.cuda.current_device() which we've set to gpu_id
            piref_model = sgl.Engine(
                model_path=args.piref_model,
                dtype=dtype,
                mem_fraction_static=requested_fraction,
                random_seed=args.seed + gpu_id,
                skip_tokenizer_init=True,
            )
            print(f"[GPU {gpu_id}] SGLang engine initialized with mem_fraction_static={requested_fraction:.3f}")
        except RuntimeError as e:
            if "Not enough memory" in str(e) or "mem-fraction-static" in str(e):
                print(f"[GPU {gpu_id}] Error: sglang ran out of memory with mem_fraction_static={requested_fraction}")
                print(f"[GPU {gpu_id}] Try reducing piref_gpu_util (current: {args.piref_gpu_util})")
                if available_gb is not None and total_gb is not None:
                    print(f"[GPU {gpu_id}] Available memory: {available_gb:.2f} GB, Total: {total_gb:.2f} GB")
            raise
        except (UnboundLocalError, NameError) as e:
            # Catch sglang's internal bug in get_available_gpu_memory
            # This happens in sglang's utils.py line 338 when free_gpu_memory is accessed before assignment
            error_str = str(e)
            if "free_gpu_memory" in error_str or "get_available_gpu_memory" in error_str:
                print(f"[GPU {gpu_id}] ERROR: Encountered sglang bug (UnboundLocalError in get_available_gpu_memory)")
                print(f"[GPU {gpu_id}] This is a bug in sglang's internal code at srt/utils.py:338, not insufficient memory.")
                if available_gb is not None and total_gb is not None:
                    print(f"[GPU {gpu_id}] GPU {gpu_id} has {available_gb:.2f} GB available memory (sufficient: {total_gb:.2f} GB total)")
                print(f"[GPU {gpu_id}] This worker will exit. Other GPUs may continue processing.")
                # Re-raise as RuntimeError with more context
                raise RuntimeError(f"sglang initialization failed due to internal bug (UnboundLocalError) on GPU {gpu_id}. "
                                 f"Memory is sufficient ({available_gb:.2f} GB available). "
                                 f"This is a bug in sglang's get_available_gpu_memory function.") from e
            raise
        except Exception as e:
            print(f"[GPU {gpu_id}] Error initializing sglang engine: {e}")
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
                
                # Generate solution
                generated_ids = generate_single_line(
                    piref_model=piref_model,
                    prompt_ids=[input_ids],
                    max_length=args.max_length,
                    block_size=args.block_size,
                    max_blocks=args.max_blocks,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_token_ids=[tokenizer.eos_token_id],
                    max_value_estimate_num_attempts=args.max_value_estimate_num_attempts,
                    classifier_model=classifier,
                    use_rejection_sampling=args.use_rejection_sampling,
                )
                
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
            # Shutdown sglang engine properly
            if piref_model is not None:
                if hasattr(piref_model, 'shutdown'):
                    piref_model.shutdown()
                    print(f"[GPU {gpu_id}] SGLang engine shutdown")
                elif hasattr(piref_model, 'close'):
                    piref_model.close()
                    print(f"[GPU {gpu_id}] SGLang engine closed")
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Error shutting down sglang engine: {e}")

        #try:
        #    # Shutdown sglang engine properly
        #    if piref_model is not None:
        #        if hasattr(piref_model, 'shutdown'):
        #            piref_model.shutdown()
        #            print(f"[GPU {gpu_id}] SGLang engine shutdown")
        #        elif hasattr(piref_model, 'close'):
        #            piref_model.close()
        #            print(f"[GPU {gpu_id}] SGLang engine closed")
        #    except Exception as e:
        #        print(f"[GPU {gpu_id}] Warning: Error shutting down sglang engine: {e}")
        

        # Clean up GPU memory
        #try:
        #    if classifier is not None:
        #        del classifier
        #    if piref_model is not None:
        #        del piref_model
        #    if tokenizer is not None:
        #        del tokenizer
        #    import gc
        #    gc.collect()
        #    if torch.cuda.is_available():
        #        torch.cuda.empty_cache()
        #        torch.cuda.synchronize()
        #except Exception as e:
        #    print(f"[GPU {gpu_id}] Warning: Error during cleanup: {e}")
        
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
        
        # Estimate memory requirements
        # sglang needs at least the fraction specified, but may need more for overhead
        # Use a more conservative estimate: reserve more than just the fraction
        sglang_reservation_gb = total_gb * args.piref_gpu_util
        # Add overhead for sglang's internal memory management (typically 10-20% more)
        sglang_overhead_gb = sglang_reservation_gb * 0.15
        sglang_total_gb = sglang_reservation_gb + sglang_overhead_gb
        
        classifier_base_memory_gb = 3.5
        max_seq_length = args.max_length
        classifier_kv_cache_per_pass_gb = (max_seq_length / 1000) * 0.4
        classifier_activation_per_pass_gb = (max_seq_length / 1000) * 0.2
        max_concurrent_passes = max(args.max_value_estimate_num_attempts, 2)
        if args.use_rejection_sampling:
            max_concurrent_passes = max(max_concurrent_passes, 3)
        classifier_peak_memory_gb = classifier_base_memory_gb + (classifier_kv_cache_per_pass_gb + classifier_activation_per_pass_gb) * max_concurrent_passes
        buffer_gb = 8.0
        required_gb = sglang_total_gb + classifier_peak_memory_gb + buffer_gb
        
        # More conservative check: ensure sglang reservation doesn't exceed 80% of available memory
        if sglang_reservation_gb > available_gb * 0.80:
            has_sufficient = False
        elif required_gb > available_gb:
            has_sufficient = False
        else:
            has_sufficient = available_gb >= required_gb
        
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
    output_path = 'processed_dataset_multigpu.jsonl'
    
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
            raise RuntimeError("No GPUs have sufficient memory. Please reduce piref_gpu_util or max_length.")
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
    
    new_dataset = datasets.Dataset.from_dict(new_dataset_dict)
    
    print(f"\nNew dataset created: {len(new_dataset)} examples")
    print(f"Column names: {new_dataset.column_names}")
    if len(new_dataset) > 0:
        print(f"\nFirst example:\n{new_dataset[0]}")
    
    # Save dataset
    output_dataset_path = "processed_openR1_100_multigpu"
    new_dataset.to_json(output_path)
    new_dataset.save_to_disk(output_dataset_path)
    print(f"\nDataset saved to {output_path} and {output_dataset_path}")
    print("Processing complete!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Auto-detect available GPUs, or specify a list like [0, 1, 2, 3]
    main_multigpu(args, num_samples=10, gpu_list=None)

