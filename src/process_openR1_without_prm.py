

from dataclasses import dataclass
import datasets
import tyro
from inference_eval_backtrack import Args
import deepseek_utils

@dataclass
class Args:
    dataset_path: str = "/data/datasets/openR1_math_220k"
    split: str = "train"
    dataset_top_n: int = 50

def main(args: Args):
    dataset = datasets.load_dataset(args.dataset_path, split=args.split)
    problems = dataset.select(range(args.dataset_top_n))["problem"]
    formatted_problems = [deepseek_utils.format_roll_in(problem) for problem in problems]


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)