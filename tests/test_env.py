"""Tests for env.debug_env.DebugEnv."""

import pytest
from unittest.mock import patch, MagicMock

from env.debug_env import DebugEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE = {
    "buggy_code": "def add(a, b):\n    return a - b\n",
    "correct_code": "def add(a, b):\n    return a + b\n",
    "test_cases": [
        "assert candidate(1, 2) == 3",
        "assert candidate(0, 0) == 0",
    ],
    "entry_point": "add",
}


def _make_exec_result(stdout="", stderr="", success=False, timeout=False):
    return {"stdout": stdout, "stderr": stderr, "success": success, "timeout": timeout}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReset:
    """reset() should return a well-formed observation."""

    @patch("env.debug_env.TOOL_REGISTRY", new_callable=dict)
    def test_observation_format(self, mock_registry):
        mock_registry["execute_code"] = MagicMock(
            return_value=_make_exec_result(stderr="AssertionError")
        )

        env = DebugEnv()
        obs = env.reset(SAMPLE)

        assert isinstance(obs, dict)
        assert obs["current_code"] == SAMPLE["buggy_code"]
        assert isinstance(obs["error_msg"], str)
        assert obs["tool_calls_used"] == 0
        assert obs["done"] is False


class TestStepExecuteCode:
    """step() with execute_code should update the observation."""

    @patch("env.debug_env.TOOL_REGISTRY", new_callable=dict)
    def test_error_msg_updated(self, mock_registry):
        exec_fn = MagicMock(
            side_effect=[
                # reset execution
                _make_exec_result(stderr="NameError: ..."),
                # step: execute_code call
                _make_exec_result(stderr="TypeError: ..."),
                # step: _tests_pass check
                _make_exec_result(stderr="TypeError: ..."),
            ]
        )
        mock_registry["execute_code"] = exec_fn

        env = DebugEnv()
        env.reset(SAMPLE)

        obs, reward, done = env.step(
            {"tool": "execute_code", "args": {"code": "print(1)"}}
        )

        assert obs["error_msg"] == "TypeError: ..."
        assert obs["tool_calls_used"] == 1


class TestStepPatchCode:
    """step() with patch_code should replace current_code."""

    @patch("env.debug_env.compute_reward", return_value=1.5)
    @patch("env.debug_env.TOOL_REGISTRY", new_callable=dict)
    def test_code_replaced(self, mock_registry, _mock_reward):
        new_code = "def add(a, b):\n    return a + b\n"

        exec_fn = MagicMock(
            side_effect=[
                # reset execution
                _make_exec_result(stderr="wrong answer"),
                # step: re-execute after patch
                _make_exec_result(stderr=""),
                # step: _tests_pass check
                _make_exec_result(stderr=""),
            ]
        )
        mock_registry["execute_code"] = exec_fn
        mock_registry["patch_code"] = MagicMock(return_value=new_code)

        env = DebugEnv()
        env.reset(SAMPLE)

        obs, reward, done = env.step(
            {
                "tool": "patch_code",
                "args": {"original": SAMPLE["buggy_code"], "patched": new_code},
            }
        )

        assert obs["current_code"] == new_code


class TestMaxToolCalls:
    """Episode must end after MAX_TOOL_CALLS tool invocations."""

    @patch("env.debug_env.compute_reward", return_value=-0.5)
    @patch("env.debug_env.TOOL_REGISTRY", new_callable=dict)
    def test_done_after_limit(self, mock_registry, _mock_reward):
        # execute_code always returns an error so tests never pass
        exec_fn = MagicMock(
            return_value=_make_exec_result(stderr="Error")
        )
        mock_registry["execute_code"] = exec_fn

        env = DebugEnv()
        env.reset(SAMPLE)

        for i in range(DebugEnv.MAX_TOOL_CALLS):
            obs, reward, done = env.step(
                {"tool": "execute_code", "args": {"code": "x"}}
            )

        assert done is True
        assert obs["tool_calls_used"] == DebugEnv.MAX_TOOL_CALLS


class TestGetTrajectory:
    """get_trajectory() must match the compute_reward schema."""

    @patch("env.debug_env.TOOL_REGISTRY", new_callable=dict)
    def test_trajectory_format(self, mock_registry):
        exec_fn = MagicMock(
            return_value=_make_exec_result(stderr="err")
        )
        mock_registry["execute_code"] = exec_fn

        env = DebugEnv()
        env.reset(SAMPLE)
        env.step({"tool": "execute_code", "args": {"code": "1"}})

        traj = env.get_trajectory()

        assert set(traj.keys()) == {
            "final_code",
            "test_cases",
            "tool_calls",
            "timeout",
            "entry_point",
        }
        assert traj["final_code"] == SAMPLE["buggy_code"]
        assert traj["test_cases"] == SAMPLE["test_cases"]
        assert traj["tool_calls"] == ["execute_code"]
        assert traj["timeout"] is False
        assert traj["entry_point"] == "add"
