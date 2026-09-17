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

# Builds instruction.md file with question, the multiple choice questions shuffled, and instructions for submission
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

# Builds the task.toml file with the task name
def build_task_toml(task_name):
    return f"""schema_version = "1.4"
    
[task]
name = "gpqa-diamond/{task_name}"

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
timeout_sec = 900.0

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

# Builds dockerfile
def build_dockerfile():
    return "FROM ubuntu:24.04\n\nWORKDIR /app\n"

# Builds the solve.sh file with the correct answer
def build_solve_sh(correct_letter):
    return f"""#!/bin/bash
echo -n "{correct_letter}" > /app/answer.txt
"""

# Builds test.sh file that checks the answer
def build_test_sh():
    return """#!/bin/bash
mkdir -p /logs/verifier

# Fail safely if answer file does not exist
if [ ! -f /app/answer.txt ]; then
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi

# Trim all whitespace, newlines, and carriage returns
actual=$(tr -d '[:space:]' < /app/answer.txt)
expected=$(tr -d '[:space:]' < /tests/expected_answer.txt)

if [ "$actual" = "$expected" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""

# Iterates through each question in gpqa diamond to generate isoalted benchmark tasks
for i, tasks in enumerate(ds["train"]):
    task_name = f"gpqa-{i:03d}"
    task_folder = base_folder / task_name
    task_folder.mkdir(parents=True, exist_ok=True)

    # Creates Harbor compliant hiearchy
    env_folder = task_folder / "environment"
    solution_folder = task_folder / "solution"
    tests_folder = task_folder / "tests"

    for folder in (env_folder, solution_folder, tests_folder):
        folder.mkdir(parents=True, exist_ok=True)

    # Aggregates choices and shuffles them with index as seed
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

    # Task root files
    (task_folder / "instruction.md").write_text(
        build_instruction(tasks["Question"], shuffled_choices),
        encoding="utf-8",
    )
    (task_folder / "task.toml").write_text(
        build_task_toml(task_name), encoding="utf-8"
    )

    # Environment
    (env_folder / "Dockerfile").write_text(
        build_dockerfile(), encoding="utf-8"
    )

    # Solution
    (solution_folder / "solve.sh").write_text(
        build_solve_sh(correct_letter), encoding="utf-8"
    )

    # Verifier tests & secrets
    (tests_folder / "test.sh").write_text(build_test_sh(), encoding="utf-8")
    (tests_folder / "expected_answer.txt").write_text(
        correct_letter, encoding="utf-8"
    )