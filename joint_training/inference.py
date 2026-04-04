import numpy as np
import re
from vllm import SamplingParams
from utils import score_trajectory, check_answer, SYSTEM_PROMPT, inverse_sigmoid


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()

def run_particle_inference(
    llm, tokenizer, prm, prm_tokenizer,
    question: str, true_answer: str,
    N: int = 8,
    max_steps: int = 10,
    method: str = "smc",          # "smc" | "particle_filter" | "bon"
    softmax_temp: float = 0.8,
    calibrator=None,
):

    system = [SYSTEM_PROMPT]
    sampling_params = SamplingParams(
        temperature=0.8, max_tokens=2048, top_p=1.0,
        stop=["\n\n", "<|eot_id|>", "\n##"],
    )

    prompt = tokenizer.apply_chat_template(
        system + [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )

    if method == "bon":
        return _run_bon(llm, tokenizer, prm, prm_tokenizer, prompt,
                        question, true_answer, N, calibrator)

    particles = []
    for _ in range(N):
        res = llm.generate(prompt, sampling_params)
        particles.append(res[0].outputs[0].text)

    scores = [
        score_trajectory(prm, prm_tokenizer, question, p, calibrator)[-1]
        for p in particles
    ]

    logits = np.array([inverse_sigmoid(s) for s in scores])
    weights = softmax(logits / softmax_temp)
    indices = np.random.choice(len(particles), size=N, replace=True, p=weights)
    particles = [particles[i] for i in indices]
    prev_scores = [scores[i] for i in indices]

    finished = []
    active = list(range(N))

    for step in range(max_steps):
        if not active:
            break

        new_particles = []
        new_scores = []
        still_active = []

        for idx in active:
            prompt_with_answer = prompt + "\n\n" + particles[idx] + "\n\n"
            res = llm.generate(prompt_with_answer, sampling_params)
            if not (res and res[0].outputs):
                continue

            extended = particles[idx] + "\n\n" + res[0].outputs[0].text
            s = score_trajectory(prm, prm_tokenizer, question, extended, calibrator)[-1]

            if tokenizer.eos_token_id in res[0].outputs[0].token_ids:
                finished.append((extended, s))
            else:
                new_particles.append(extended)
                new_scores.append(s)
                still_active.append(idx)

        if not new_particles:
            break

        if method == "smc":
            # SMC: incremental weights = logit(new) - logit(old)
            old = np.array([inverse_sigmoid(prev_scores[i]) for i in still_active])
            new = np.array([inverse_sigmoid(s) for s in new_scores])
            log_weights = (new - old) / softmax_temp
        else:
            # Particle filter (Rollout Roulette): absolute weights
            log_weights = np.array([inverse_sigmoid(s) for s in new_scores]) / softmax_temp

        weights = softmax(log_weights)
        M = len(new_particles)
        indices = np.random.choice(M, size=M, replace=True, p=weights)

        particles = {i: new_particles[indices[j]] for j, i in enumerate(range(M))}
        prev_scores_map = {i: new_scores[indices[j]] for j, i in enumerate(range(M))}

        # Rebuild flat lists
        active = list(range(M))
        particles_list = [particles[i] for i in active]
        prev_scores = [prev_scores_map[i] for i in active]
        particles = particles_list

    # Force-finish any remaining active particles
    for p in particles:
        forced_prompt = prompt + p + "\n\nTherefore, the final answer is \\boxed{"
        res = llm.generate(forced_prompt, SamplingParams(
            temperature=0.8, max_tokens=256, top_p=1.0))
        full = p + "\n\nTherefore, the final answer is \\boxed{" + res[0].outputs[0].text
        s = score_trajectory(prm, prm_tokenizer, question, full, calibrator)[-1]
        finished.append((full, s))

    trajectories = []
    for text, prm_score in finished:
        trajectories.append({
            "text": text,
            "correct": check_answer(text, true_answer),
            "prm_score": prm_score,
        })

    n_correct = sum(t["correct"] for t in trajectories)
    return {
        "question": question,
        "true_answer": true_answer,
        "trajectories": trajectories,
        "pass_rate": n_correct / max(len(trajectories), 1),
    }


def _run_bon(llm, tokenizer, prm, prm_tokenizer, prompt,
             question, true_answer, N, calibrator):
    """Best-of-N: generate N full answers, score, pick best."""
    full_params = SamplingParams(
        temperature=0.8, max_tokens=4096, top_p=1.0,
        stop=["<|eot_id|>"],
    )
    trajectories = []
    for _ in range(N):
        res = llm.generate(prompt, full_params)
        text = res[0].outputs[0].text
        scores = score_trajectory(prm, prm_tokenizer, question, text, calibrator)
        trajectories.append({
            "text": text,
            "correct": check_answer(text, true_answer),
            "prm_score": scores[-1] if scores else 0.0,
        })
    n_correct = sum(t["correct"] for t in trajectories)
    return {
        "question": question,
        "true_answer": true_answer,
        "trajectories": trajectories,
        "pass_rate": n_correct / max(len(trajectories), 1),
    }


def run_inference(llm, tokenizer, prm, prm_tokenizer, dataset, *,
                  N=8, max_steps=10, method="smc", calibrator=None):
    """Run inference on a list of problems. Returns list of result dicts."""
    results = []
    for i, item in enumerate(dataset):
        print(f"  [{method}] Problem {i+1}/{len(dataset)}", end="\r")
        result = run_particle_inference(
            llm, tokenizer, prm, prm_tokenizer,
            item["problem"], item["answer"],
            N=N, max_steps=max_steps, method=method,
            calibrator=calibrator,
        )
        results.append(result)
    print()
    return results
