"""Download and cache the HumanEval dataset."""

import os
from datasets import load_dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_humaneval():
    """Load HumanEval dataset, caching to data/cache/."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dataset = load_dataset("openai_humaneval", split="test", cache_dir=CACHE_DIR)
    return dataset
