import re
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
from math_verify import parse, verify as math_verify_fn, LatexExtractionConfig, ExprExtractionConfig
 
  
def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        i += 1
    return text[start:i-1].strip() if depth == 0 else None
 
 
def check_answer(generated: str, ground_truth: str) -> bool:
    gen_ans = extract_boxed(generated)
    if gen_ans is None:
        return False
    try:
        cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        gt_parsed = parse(ground_truth, extraction_config=cfg)
        gen_parsed = parse(gen_ans, extraction_config=cfg)
        if gt_parsed and gen_parsed:
            return bool(math_verify_fn(gt_parsed[0], gen_parsed[0]))
    except Exception:
        pass
    # Fallback: normalized string comparison
    def norm(s):
        s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
        return re.sub(r'\s+', '', s).lower().strip()
    return norm(gen_ans) == norm(ground_truth)
 
  
def make_step_rewards(logits, token_masks):
    """Extract per-step rewards from PRM logits at <extra_0> positions."""
    probs = F.softmax(logits, dim=-1) * token_masks.unsqueeze(-1)
    results = []
    for i in range(probs.size(0)):
        sample = probs[i]
        positive = sample[sample != 0].view(-1, 2)[:, 1]
        results.append(positive.cpu().tolist())
    return results
 
 
def score_trajectory(prm, prm_tokenizer, question: str, answer: str,
                     calibrator=None) -> list[float]:

    steps = re.sub(r'\n+', '\n', answer).split("\n\n")
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conversation = prm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = prm_tokenizer.encode(conversation, return_tensors="pt").to(prm.device)
    with torch.no_grad():
        outputs = prm(input_ids=input_ids)
 
    step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_rewards = make_step_rewards(outputs[0], token_masks)[0]
 
    if calibrator is not None:
        step_rewards = [calibrator(s) for s in step_rewards]
    return step_rewards
 
  
def load_math500(n_train: int = 100, seed: int = 42):

    ds = hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))
    train_idx, eval_idx = indices[:n_train], indices[n_train:]
 
    def to_list(idx):
        return [
            {"problem": ds[int(i)]["problem"],
             "answer": ds[int(i)]["answer"],
             "level": ds[int(i)].get("level", ""),
             "subject": ds[int(i)].get("subject", "")}
            for i in idx
        ]
    return to_list(train_idx), to_list(eval_idx)
 
 
# PRM calibration isotonic regression
 
class PRMCalibrator: 
    ## Claude's suggestion: using isotonic regression as a lightweight alternative to the full quantile-regression
    ## we can swithc to quantile regression if this was not good.
 
    def __init__(self):
        self.raw_scores: list[float] = []
        self.outcomes: list[int] = []   # 1 = correct, 0 = wrong
        self._iso = None
 
    def update(self, scores: list[float], outcomes: list[int]):
        self.raw_scores.extend(scores)
        self.outcomes.extend(outcomes)
 
    def fit(self):
        ## fit isotonic regression 
        if len(self.raw_scores) < 10:
            return
        from sklearn.isotonic import IsotonicRegression
        X = np.array(self.raw_scores)
        y = np.array(self.outcomes, dtype=float)
        self._iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        self._iso.fit(X, y)
 
    def __call__(self, score: float) -> float:
        if self._iso is None:
            return score
        return float(self._iso.predict([score])[0])
 
  
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\n"
        "Provide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\n"
        "Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n"
        "...\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem."
    ),
}
 
 
def inverse_sigmoid(x: float) -> float:
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return float(np.log(x) - np.log(1 - x))
 
