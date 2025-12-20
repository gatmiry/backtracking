import datasets
import transformers
from inference_eval_backtrack import get_eos_token_id
from typing import List
import deepseek_utils
import accuracy_utils
import classifier_lib
import tyro
from inference_eval_backtrack import Args
import torch
import sglang as sgl
from inference_eval_backtrack import generate_single_line
class DeepseekStylePreprocessor:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = get_eos_token_id(tokenizer)

    def process_problem(self, problem_batch: List[str]) -> List[int]:
        formatted_problems = [deepseek_utils.format_roll_in(problem) for problem in problem_batch]
        input_ids = self.tokenizer(formatted_problems, add_special_tokens=False)["input_ids"]
        print(f"Input IDs: {input_ids}")
        return input_ids

    def process_response(self, response_batch: List[str], ground_truth_batch: str) -> List[int]:
        generated_texts = self.tokenizer.batch_decode(response_batch, skip_special_tokens=False)
        generated_solutions = [deepseek_utils.remove_thinking_text(text) for text in generated_texts]
        rewards = [accuracy_utils.math_verify_check(ground_truth, generated_solution) for ground_truth, generated_solution in zip(ground_truth_batch, generated_solutions)]
        formatted_responses = [deepseek_utils.format_roll_out(response) for response in response_batch]
        input_ids = self.tokenizer(formatted_responses, add_special_tokens=False)["input_ids"]
        return input_ids

if __name__ == "__main__":
    # Example usage
    args =tyro.cli(Args)
    device = 'cuda:0'
    dataset = datasets.load_from_disk('/data/datasets/openR1_math_220k')
    dataset = dataset.select(range(2))
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Column names: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"\nFirst example:\n{dataset[0]}")
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    preprocessor = DeepseekStylePreprocessor(tokenizer)
    piref_model = sgl.Engine(
        model_path=args.piref_model,
        dtype=dtype,
        device=device,
        mem_fraction_static=args.piref_gpu_util,
        random_seed=args.seed,
        skip_tokenizer_init=True,
    )
    model_loading_kwargs = dict(attn_implementation=args.attention_impl, torch_dtype=dtype, use_cache=True)
    classifier = classifier_lib.Qwen2ForClassifier.from_pretrained(args.classifier_ckpt_path, **model_loading_kwargs).to(device)
    #print('piref model device: ', piref_model.device)
    print('classifier model device: ', classifier.device)
    model = piref_model.get_model()
    print('model device: ', model.device)

    def generate_solutions(examples):
        #print('length of examples is: ', len(examples["problem"]))
        processed_examples = preprocessor.process_problem(examples["problem"])
        generated_solutions = []
        processed_answers = []
        #print('length of processed_examples is: ', len(processed_examples))
        for _, processed_example in enumerate(processed_examples):
            #print(f"the prompt was: {examples['problem'][counter]}")
            #print(f"Generating solution for the ids of this prompt which are: {processed_example}")
            generated_ids = generate_single_line(
                piref_model=piref_model,
                prompt_ids=[processed_example],
                max_length=args.max_length,
                block_size=args.block_size,
                max_blocks=args.max_blocks,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_token_ids=[tokenizer.eos_token_id],
                max_value_estimate_num_attempts=args.max_value_estimate_num_attempts, 
                classifier_model=classifier, 
                use_rejection_sampling=False)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
            generated_solution = deepseek_utils.remove_thinking_text(generated_text)
            processed_answer = accuracy_utils.process_sample(generated_solution)
            generated_solutions.append(generated_solution)
            processed_answers.append(processed_answer)
            
        return {'problems': examples["problem"], 
            #'generated_solutions': generated_solutions, 
            'processed_answers': processed_answers}
    
    # Process dataset using map
    processed_dataset = dataset.map(generate_solutions, batched=True, batch_size=2)
    print(f"\nProcessed dataset: {len(processed_dataset)} examples")
    print(f"Column names: {processed_dataset.column_names}")
    if len(processed_dataset) > 0:
        print(f"\nFirst example:\n{processed_dataset[0]}")
    
    processed_dataset.to_json('processed_dataset.jsonl')

    