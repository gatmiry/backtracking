import datasets

dataset = datasets.load_dataset("open-r1/OpenR1-Math-220k", split="train")
dataset.save_to_disk("./data/OpenR1-Math-220k")