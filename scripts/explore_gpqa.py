from dotenv import load_dotenv
from datasets import load_dataset
import os

load_dotenv()

ds = load_dataset(
    "Idavidrein/gpqa",
    "gpqa_diamond",
    token = os.environ["HF_TOKEN"],
)
print(ds)
print(ds["train"][0])