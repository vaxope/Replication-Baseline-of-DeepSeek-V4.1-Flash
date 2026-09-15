from dotenv import load_dotenv
from datasets import load_dataset
import os
from pathlib import Path

load_dotenv()

ds = load_dataset(
    "Idavidrein/gpqa",
    "gpqa_diamond",
    token = os.environ["HF_TOKEN"],
)

base_folder = Path("tasks/gpqa_diamond")

for i, tasks in enumerate(ds["train"]):
    task_name = f"gpqa-{i:03d}"
    task_folder = base_folder / task_name
    task_folder.mkdir(parents=True, exist_ok=True)

    instruction_path = task_folder / "instruction.md"
    with open(instruction_path, "w", encoding="utf-8") as file:
        file.write(tasks["Question"])