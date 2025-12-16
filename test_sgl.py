import os
from generate_backtrack_batch import generate_backtrack_batch
import classifier_lib
import deepseek_utils
import accuracy_utils
import ujson as json
# CRITICAL: Set CUDA_VISIBLE_DEVICES before any CUDA-related imports
# This must happen before importing torch, transformers, or sglang
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
    if result.returncode == 0:
        num_gpus = len(result.stdout.strip().split('\n'))
        if num_gpus > 1:
            # Use GPUs 1, 2 (2 GPUs) since vocab size must be divisible by tp_size
            # Vocab size 151936 is divisible by 2, 4, 8, etc. but not 3
            gpus = ','.join(str(i) for i in range(1, 3))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            print(f"Using GPUs: {gpus} (excluding GPU 0)")

from transformers import AutoTokenizer
import sglang as sgl
import torch
import datasets
from process_openR1 import DeepseekStylePreprocessor
import deepseek_utils
import accuracy_utils

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    eos_token_id = tokenizer.convert_tokens_to_ids(['<｜end▁of▁sentence｜>'])[0]

    seed = 1337
    model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Find valid tensor parallelism size that divides vocabulary size
    # Vocab size for this model is 151936, which is divisible by: 1, 2, 4, 8, 16, 32, 64, etc.
    # IMPORTANT: Get device count AFTER CUDA_VISIBLE_DEVICES is set
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    vocab_size = len(tokenizer)
    
    # Determine how many GPUs we actually set in CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        num_visible_gpus = len(cuda_visible.split(','))
    else:
        num_visible_gpus = available_gpus
    
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"Available GPUs (after CUDA_VISIBLE_DEVICES): {available_gpus}")
    print(f"Number of visible GPUs: {num_visible_gpus}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Find the largest valid tp_size <= num_visible_gpus that divides vocab_size
    # Use num_visible_gpus to ensure we don't exceed the GPUs we actually set
    max_tp_size = min(available_gpus, num_visible_gpus)
    tp_size = 1
    for candidate in range(max_tp_size, 0, -1):
        if vocab_size % candidate == 0:
            tp_size = candidate
            break
    
    print(f"Using {tp_size} GPU(s) with tensor parallelism (vocab_size={vocab_size} must be divisible by tp_size)")
    
    engine = sgl.Engine(model_path=model_path, dtype=dtype, device=device, tp_size=tp_size, mem_fraction_static=0.4, random_seed=seed, skip_tokenizer_init=True)
    max_length = 16384
    block_size = 4096
    num_blocks = 8
    max_length_without_prm = 16384
    dataset_top_n = 50
    sampling_params = [{
        'temperature': 0.6, 
        'top_p': 0.9,
        'skip_special_tokens': False,
        'stop_token_ids': [eos_token_id],
        'max_new_tokens': max_length_without_prm
    }] * dataset_top_n

    dataset = datasets.load_from_disk('/data/datasets/openR1_math_220k')
    problems = dataset[0:dataset_top_n]['problem']
    answers = dataset[0:dataset_top_n]['answer']
    
    # Format problems with the required special tokens
    formatted_problems = [deepseek_utils.format_roll_in(problem) for problem in problems]
    print('formatted_problems: ', formatted_problems)
    # Tokenize the formatted problems (no padding - each sequence is independent)
    tokenized = tokenizer(formatted_problems, add_special_tokens=False, padding=False)
    input_ids = tokenized['input_ids']
    # Ensure input_ids is a list of lists of integers
    input_ids = [[int(token_id) for token_id in seq] for seq in input_ids]
    
    # sampling_params is already a list of 50 (one per prompt)
    
    #outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    classifier_model = classifier_lib.Qwen2ForClassifier.from_pretrained("VGS-AI/DeepSeek-VM-1.5B").to(device)
    
    ## generating without the prm
    base_model_outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    base_generated_tokens = [output['output_ids'] for output in base_model_outputs]
    base_generated_texts = tokenizer.batch_decode(base_generated_tokens, skip_special_tokens=False)
    base_generated_solutions = [deepseek_utils.remove_thinking_text(text) for text in base_generated_texts]
    base_processed_answers = [accuracy_utils.process_sample(solution) for solution in base_generated_solutions]
    base_rewards = [
        accuracy_utils.math_verify_check(answers[i], base_processed_answers[i])
        for i in range(dataset_top_n)
    ]

    max_value_estimate_num_attempts = 6
    print('eos token id: ', eos_token_id)
    outputs = generate_backtrack_batch(engine, input_ids, max_length, block_size, num_blocks, 0.6, 0.9, [eos_token_id], max_value_estimate_num_attempts, classifier_model, False)
    generated_texts = tokenizer.batch_decode([output['output_ids'] for output in outputs], skip_special_tokens=False)
    generated_solutions = [deepseek_utils.remove_thinking_text(text) for text in generated_texts]
    processed_answers = [accuracy_utils.process_sample(solution) for solution in generated_solutions]
    rewards = [
        accuracy_utils.math_verify_check(ground_truth, processed)
        for ground_truth, processed in zip(answers, processed_answers)
    ]

    records = []
    for i in range(len(problems)):
        record = {
            "problem": problems[i],
            "processed_answer": processed_answers[i],
            "reward": rewards[i],
            "processed_answer_without_prm": base_processed_answers[i],
            "reward_without_prm": base_rewards[i],
            "generate_solution": generated_solutions[i],
            "generated_solution_without_prm": base_generated_solutions[i],
        }
        records.append(record)

    output_path = 'outputs/test_sgl_blocksize_4096_mve_6.jsonl'
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
        f.flush()

    from datasets import Dataset
    processed_dataset = Dataset.from_list(records)
    processed_dataset.save_to_disk('outputs/test_sgl_processed_blocksize_4096_mve_6')

    #print(generated_texts[:5])
    #print(processed_answers[:5])
    #print(rewards[:5])