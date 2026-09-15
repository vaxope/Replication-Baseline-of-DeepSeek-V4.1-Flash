from dotenv import load_dotenv
from datasets import load_dataset
import os
from pathlib import Path
import random

load_dotenv()

ds = load_dataset(
    "Idavidrein/gpqa",
    "gpqa_diamond",
    token = os.environ["HF_TOKEN"],
)

base_folder = Path("tasks/gpqa_diamond")

def build_instruction(question, shuffled_choices):
    choices_text = "\n".join(
        f"{letter}) {choice.strip()}"
        for letter, choice in zip("ABCD", shuffled_choices)
    )

    return f"""Answer the following multiple-choice science question.

{question.strip()}

{choices_text}

Think through the problem carefully. When you have determined the correct \
answer, write ONLY the single letter (A, B, C, or D) to a file called \
/app/answer.txt, with no extra text, punctuation, or explanation in the file.

Example: if you believe the answer is C, run:
echo -n "C" > /app/answer.txt

Then mark the task complete.
"""

def build_task_toml(task_name):
    return f"""schema_version = "1.4"
    
[task]
name = "gpqa-diamong/{task_name}"

version = "1.0.0"
authors = []
keywords = []

[metadata]
author_name = "Replication Baseline"
difficulty = "hard"
category = "science"
tags = ["gpqa", "multiple-choice"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 10240
gpus = 0
mcp_servers = []

[verifier.env]

[solution.env]
"""

def build_dockerfile():
    return "FROM ubuntu:24.04\n\nWORKDIR /app\n"

def build_solve_sh(correct_letter):
    return ""

def build_test_sh():
    pass

for i, tasks in enumerate(ds["train"]):
    task_name = f"gpqa-{i:03d}"
    task_folder = base_folder / task_name
    task_folder.mkdir(parents=True, exist_ok=True)

    choices = [
        tasks["Correct Answer"],
        tasks["Incorrect Answer 1"],
        tasks["Incorrect Answer 2"],
        tasks["Incorrect Answer 3"],
    ]

    order = [0, 1, 2, 3]
    rng = random.Random(i)
    rng.shuffle(order)

    shuffled_choices = [choices[pos] for pos in order]
    correct_index = order.index(0)
    correct_letter = "ABCD"[correct_index]

    # Writes the instruction.md file containing the question
    instruction_path = task_folder / "instruction.md"
    with open(instruction_path, "w", encoding="utf-8") as file:
        file.write(build_instruction(tasks["Question"], shuffled_choices))