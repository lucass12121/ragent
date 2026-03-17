"""DebugDataset: pairs of buggy and correct code from HumanEval."""

from data.download import load_humaneval
from data.bug_injector import BugInjector


class DebugDataset:
    """Dataset of (buggy_code, correct_code, test_cases, entry_point) items."""

    def __init__(self, split: str = "train"):
        raw = load_humaneval()
        n = len(raw)
        split_idx = int(n * 0.8)

        if split == "train":
            self._items = [raw[i] for i in range(0, split_idx)]
        else:
            self._items = [raw[i] for i in range(split_idx, n)]

        self._injector = BugInjector()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        correct_code = item["canonical_solution"]
        prompt = item["prompt"]
        full_correct = prompt + correct_code

        buggy_code = self._injector.inject(full_correct)
        test_cases = self._parse_tests(item["test"])

        return {
            "buggy_code": buggy_code,
            "correct_code": full_correct,
            "test_cases": test_cases,
            "entry_point": item["entry_point"],
        }

    @staticmethod
    def _parse_tests(test_str: str) -> list:
        """Extract individual assert statements from the test code string."""
        tests = []
        for line in test_str.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert"):
                tests.append(stripped)
        return tests
