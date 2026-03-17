import pytest
from reward.reward_fn import run_tests, compute_reward


CORRECT_CODE = "def add(a, b):\n    return a + b"
WRONG_CODE = "def add(a, b):\n    return a - b"
SYNTAX_ERROR_CODE = "def add(a, b)\n    return a + b"

TEST_CASES = [
    {"input": "1, 2", "output": "3"},
    {"input": "0, 0", "output": "0"},
    {"input": "-1, 1", "output": "0"},
    {"input": "10, 20", "output": "30"},
    {"input": "100, 200", "output": "300"},
]


class TestRunTests:
    def test_correct_code_passes_all(self):
        passed = run_tests(CORRECT_CODE, TEST_CASES, "add")
        assert passed == len(TEST_CASES)

    def test_wrong_code_passes_none(self):
        passed = run_tests(WRONG_CODE, TEST_CASES, "add")
        # Only (0,0) passes for a-b
        assert passed < len(TEST_CASES)

    def test_syntax_error_passes_none(self):
        passed = run_tests(SYNTAX_ERROR_CODE, TEST_CASES, "add")
        assert passed == 0


class TestComputeReward:
    def test_correct_code_high_reward(self):
        trajectory = {
            "final_code": CORRECT_CODE,
            "test_cases": TEST_CASES,
            "tool_calls": ["read_file"],
            "timeout": False,
            "entry_point": "add",
        }
        reward = compute_reward(trajectory)
        assert reward > 1.0

    def test_wrong_code_low_reward(self):
        trajectory = {
            "final_code": SYNTAX_ERROR_CODE,
            "test_cases": TEST_CASES,
            "tool_calls": ["read_file", "edit_file", "run_code"] * 5,
            "timeout": False,
            "entry_point": "add",
        }
        reward = compute_reward(trajectory)
        assert reward < 0

    def test_more_tool_calls_lower_reward(self):
        base = {
            "final_code": CORRECT_CODE,
            "test_cases": TEST_CASES,
            "timeout": False,
            "entry_point": "add",
        }
        r_few = compute_reward({**base, "tool_calls": ["a"]})
        r_many = compute_reward({**base, "tool_calls": ["a"] * 20})
        assert r_few > r_many

    def test_timeout_penalty(self):
        base = {
            "final_code": CORRECT_CODE,
            "test_cases": TEST_CASES,
            "tool_calls": [],
            "entry_point": "add",
        }
        r_no_timeout = compute_reward({**base, "timeout": False})
        r_timeout = compute_reward({**base, "timeout": True})
        assert r_no_timeout - r_timeout == pytest.approx(0.5)

    def test_reward_clipped(self):
        # Many tool calls should not go below -1.0
        trajectory = {
            "final_code": SYNTAX_ERROR_CODE,
            "test_cases": TEST_CASES,
            "tool_calls": ["x"] * 200,
            "timeout": True,
            "entry_point": "add",
        }
        reward = compute_reward(trajectory)
        assert reward >= -1.0
        assert reward <= 3.0
