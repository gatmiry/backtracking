## inference time scaling → filtering questions based on success rate → fine-tune generator → calibrating prm.
## python train_loop.py --num_iterations 5 --N 8 --method smc/particle_filter/bon

import argparse
import json
import os
import gc
import shutil

import numpy as np
import torch
from vllm import LLM
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from utils import load_math500, SYSTEM_PROMPT, PRMCalibrator
from inference import run_inference

def filter_by_difficulty(results, low=0.1, high=0.9):
    filtered = []
    for r in results:
        if low < r["pass_rate"] < high:
            filtered.append(r)
    return filtered


## full on Claude for the SFT LoRA code

# ── SFT data preparation ────────────────────────────────────────────────────

def prepare_sft_data(results, tokenizer):
    """From filtered results, collect correct trajectories as SFT examples.

    Each example is formatted as a chat conversation:
        system: [prompt]
        user:   [question]
        assistant: [correct trajectory]
    """
    examples = []
    for r in results:
        for traj in r["trajectories"]:
            if traj["correct"]:
                messages = [
                    SYSTEM_PROMPT,
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": traj["text"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                examples.append({"text": text})
    return examples


# ── LoRA fine-tuning ─────────────────────────────────────────────────────────

def finetune_generator(model_path: str, sft_examples: list[dict],
                       iteration: int, output_dir: str = "checkpoints",
                       lr: float = 2e-4, epochs: int = 2, batch_size: int = 4):
    """Fine-tune the generator with LoRA, merge, and save.

    Returns the path to the merged model for the next iteration.
    """
    save_path = os.path.join(output_dir, f"iter_{iteration}")

    print(f"  Fine-tuning on {len(sft_examples)} examples → {save_path}")

    # Load model for training (transformers, not vLLM)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = Dataset.from_list(sft_examples)

    training_args = SFTConfig(
        output_dir=save_path + "_lora",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_accumulation_steps=2,
        max_seq_length=4096,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()

    # Merge LoRA into base and save
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Cleanup
    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return save_path


# ── PRM calibration ──────────────────────────────────────────────────────────

def update_calibrator(calibrator: PRMCalibrator, results: list[dict]):
    """Feed trajectory-level (prm_score, correct) pairs into the calibrator."""
    scores, outcomes = [], []
    for r in results:
        for traj in r["trajectories"]:
            scores.append(traj["prm_score"])
            outcomes.append(int(traj["correct"]))
    calibrator.update(scores, outcomes)
    calibrator.fit()
    print(f"  Calibrator updated with {len(scores)} samples "
          f"(total: {len(calibrator.raw_scores)})")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics from inference results."""
    pass_at_1 = []  # best-by-PRM is correct
    pass_at_n = []  # at least one correct
    pass_rates = []

    for r in results:
        trajs = r["trajectories"]
        if not trajs:
            pass_at_1.append(0)
            pass_at_n.append(0)
            pass_rates.append(0.0)
            continue
        # pass@N
        any_correct = int(any(t["correct"] for t in trajs))
        pass_at_n.append(any_correct)
        # pass@1 (PRM-selected)
        best = max(trajs, key=lambda t: t["prm_score"])
        pass_at_1.append(int(best["correct"]))
        pass_rates.append(r["pass_rate"])

    return {
        "pass@1": np.mean(pass_at_1),
        "pass@N": np.mean(pass_at_n),
        "avg_pass_rate": np.mean(pass_rates),
    }


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Self-improvement training loop")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prm", type=str, default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--N", type=int, default=8,
                        help="Number of trajectories per question")
    parser.add_argument("--method", type=str, default="smc",
                        choices=["smc", "particle_filter", "bon"])
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--low_threshold", type=float, default=0.1,
                        help="Min pass rate to include question for training")
    parser.add_argument("--high_threshold", type=float, default=0.9,
                        help="Max pass rate to include question for training")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_file", type=str, default="metrics.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Load dataset ─────────────────────────────────────────────────────
    print("Loading MATH-500 dataset...")
    train_data, eval_data = load_math500(n_train=args.n_train, seed=args.seed)
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # ── Load PRM (stays on GPU:1 throughout) ─────────────────────────────
    print("Loading PRM...")
    prm = AutoModel.from_pretrained(
        args.prm, device_map="cuda:1",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm)

    calibrator = PRMCalibrator()
    model_path = args.model
    all_metrics = []

    for iteration in range(args.num_iterations):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}")
        print(f"{'='*60}")

        # ── 1. Load generator ────────────────────────────────────────────
        print("Loading generator with vLLM...")
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.8,
            seed=args.seed + iteration,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_model_len=4096,
        )
        tokenizer = llm.get_tokenizer()

        # ── 2. Inference on train set ────────────────────────────────────
        print("Running inference on train set...")
        cal = calibrator if calibrator._iso is not None else None
        train_results = run_inference(
            llm, tokenizer, prm, prm_tokenizer, train_data,
            N=args.N, max_steps=args.max_steps, method=args.method,
            calibrator=cal,
        )
        train_metrics = compute_metrics(train_results)
        print(f"  Train metrics: {train_metrics}")

        # ── 3. Inference on eval set ─────────────────────────────────────
        print("Running inference on eval set...")
        eval_results = run_inference(
            llm, tokenizer, prm, prm_tokenizer, eval_data,
            N=args.N, max_steps=args.max_steps, method=args.method,
            calibrator=cal,
        )
        eval_metrics = compute_metrics(eval_results)
        print(f"  Eval metrics:  {eval_metrics}")

        # Free vLLM before fine-tuning (needs the GPU)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # ── 4. Filter questions by difficulty ────────────────────────────
        filtered = filter_by_difficulty(
            train_results, low=args.low_threshold, high=args.high_threshold
        )
        print(f"  Filtered: {len(filtered)}/{len(train_results)} questions "
              f"(pass rate in ({args.low_threshold}, {args.high_threshold}))")

        # ── 5. Prepare SFT data and fine-tune ────────────────────────────
        gen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token
        sft_examples = prepare_sft_data(filtered, gen_tokenizer)
        print(f"  SFT examples: {len(sft_examples)}")

        if sft_examples:
            model_path = finetune_generator(
                model_path, sft_examples, iteration,
                output_dir=args.output_dir,
                lr=args.lr, epochs=args.sft_epochs,
            )
        else:
            print("  No SFT examples — skipping fine-tuning this round.")

        # ── 6. Update PRM calibrator ─────────────────────────────────────
        print("Updating PRM calibrator...")
        update_calibrator(calibrator, train_results)

        # ── 7. Log ───────────────────────────────────────────────────────
        record = {
            "iteration": iteration,
            "model_path": model_path,
            "method": args.method,
            "N": args.N,
            "n_filtered_questions": len(filtered),
            "n_sft_examples": len(sft_examples),
            "train": train_metrics,
            "eval": eval_metrics,
        }
        all_metrics.append(record)
        with open(args.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Logged to {args.log_file}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    for m in all_metrics:
        print(f"  Iter {m['iteration']}: "
              f"train pass@1={m['train']['pass@1']:.3f}, "
              f"eval pass@1={m['eval']['pass@1']:.3f}, "
              f"eval pass@N={m['eval']['pass@N']:.3f}")


if __name__ == "__main__":
    main()
