import random

random.seed(42)
sample_ids = random.sample(range(198), 35)
task_names = [f"gpqa-{i:03d}" for i in sorted(sample_ids)]
print(" ".join(f"--include-task-name {n}" for n in task_names))