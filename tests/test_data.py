"""Tests for data loading and bug injection."""

import pytest
from data.bug_injector import BugInjector
from data.dataset import DebugDataset


# ── BugInjector tests ──────────────────────────────────────────────

class TestBugInjector:
    def setup_method(self):
        self.injector = BugInjector()
        self.sample_code = (
            "def add(a, b):\n"
            "    result = a + b\n"
            "    items = [1, 2, 3]\n"
            "    val = items[0]\n"
            "    return result\n"
        )

    def test_index_error_injection(self):
        result = self.injector.inject(self.sample_code, bug_type="index_error")
        assert result != self.sample_code, "index_error injection should modify the code"

    def test_type_error_injection(self):
        result = self.injector.inject(self.sample_code, bug_type="type_error")
        assert result != self.sample_code, "type_error injection should modify the code"
        assert '"result: "' in result, "type_error should inject string concatenation"

    def test_logic_error_injection(self):
        result = self.injector.inject(self.sample_code, bug_type="logic_error")
        assert result != self.sample_code, "logic_error injection should modify the code"

    def test_random_injection(self):
        result = self.injector.inject(self.sample_code, bug_type="random")
        assert result != self.sample_code, "random injection should modify the code"


# ── DebugDataset tests ─────────────────────────────────────────────

class TestDebugDataset:
    @pytest.fixture(autouse=True, scope="class")
    def _load_dataset(self, request):
        request.cls.dataset = DebugDataset(split="train")

    def test_dataset_length(self):
        assert len(self.dataset) > 0, "Dataset should not be empty"

    def test_first_item_format(self):
        item = self.dataset[0]
        assert isinstance(item, dict)
        for key in ("buggy_code", "correct_code", "test_cases", "entry_point"):
            assert key in item, f"Missing key: {key}"
        assert isinstance(item["buggy_code"], str)
        assert isinstance(item["correct_code"], str)
        assert isinstance(item["test_cases"], list)
        assert isinstance(item["entry_point"], str)
        assert item["buggy_code"] != item["correct_code"], "Buggy code should differ from correct code"

    def test_train_test_split(self):
        train_ds = self.dataset
        test_ds = DebugDataset(split="test")
        assert len(train_ds) + len(test_ds) == 164  # HumanEval has 164 problems
