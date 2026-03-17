import subprocess
import sys
import tempfile
import os


def run_tests(code: str, test_cases: list, entry_point: str) -> int:
    """Execute code and run test cases, returning the number of passed tests.

    Supports two test_cases formats:
      - list of dicts with 'input'/'output' keys
      - list of assert-statement strings (from DebugDataset)
    """
    passed = 0
    for tc in test_cases:
        if isinstance(tc, dict):
            script = (
                f"{code}\n"
                f"import json\n"
                f"result = {entry_point}({tc['input']})\n"
                f"expected = {tc['output']}\n"
                f"assert result == expected, f'Got {{result}}, expected {{expected}}'\n"
            )
        else:
            # tc is an assert-string like "assert func(1) == 2"
            # HumanEval tests reference `candidate`; alias it
            script = (
                f"{code}\n"
                f"candidate = {entry_point}\n"
                f"{tc}\n"
            )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                tmp_path = f.name
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                timeout=3,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return passed


def compute_reward(trajectory: dict) -> float:
    """Compute reward for a debugging trajectory.

    Reward rules:
      - Code executes without error:   +1.0
      - Per passed test case:          +0.2 (max +1.0)
      - Per tool call:                 -0.05
      - Timeout:                       -0.5
      - Final reward clipped to [-1.0, 3.0]
    """
    reward = 0.0

    final_code = trajectory["final_code"]
    test_cases = trajectory["test_cases"]
    tool_calls = trajectory.get("tool_calls", [])
    timeout = trajectory.get("timeout", False)
    entry_point = trajectory["entry_point"]

    # Check if code executes without error
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(final_code)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0:
            reward += 1.0
    except Exception:
        pass
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Test case reward
    passed = run_tests(final_code, test_cases, entry_point)
    reward += min(passed * 0.2, 1.0)

    # Penalties
    reward -= len(tool_calls) * 0.05
    if timeout:
        reward -= 0.5

    # Clip
    reward = max(-1.0, min(3.0, reward))
    return reward
