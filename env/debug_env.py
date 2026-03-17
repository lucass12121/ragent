"""DebugEnv: simulates a single debugging episode."""

from tools import TOOL_REGISTRY
from reward.reward_fn import compute_reward


class DebugEnv:
    """Environment for a code-debugging episode."""

    MAX_TOOL_CALLS = 10

    def __init__(self):
        self.current_code: str = ""
        self.tool_call_history: list = []
        self.done: bool = False
        self.sample: dict = {}
        self._last_error: str = ""
        self._timeout: bool = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self, sample: dict) -> dict:
        """Start a new episode from *sample* (one item from DebugDataset).

        Returns an initial observation dict.
        """
        self.sample = sample
        self.current_code = sample["buggy_code"]
        self.tool_call_history = []
        self.done = False
        self._timeout = False

        # Run the buggy code to obtain the initial error message
        exec_result = TOOL_REGISTRY["execute_code"](self.current_code)
        self._last_error = exec_result["stderr"]
        if exec_result["timeout"]:
            self._timeout = True

        return self._make_observation()

    def step(self, action: dict) -> tuple:
        """Execute one agent action.

        Args:
            action: {"tool": str, "args": dict}

        Returns:
            (observation, reward, done)
        """
        if self.done:
            return self._make_observation(), 0.0, True

        tool_name = action["tool"]
        args = action["args"]

        # Call the tool
        tool_fn = TOOL_REGISTRY[tool_name]
        result = tool_fn(**args)

        # Record the call
        self.tool_call_history.append(
            {"tool": tool_name, "args": args, "result": result}
        )

        # Update internal state depending on the tool
        if tool_name == "execute_code":
            self._last_error = result["stderr"]
            if result["timeout"]:
                self._timeout = True

        elif tool_name == "patch_code":
            # patch_code returns the (validated) code string
            self.current_code = result
            # Re-execute to refresh the error message
            exec_result = TOOL_REGISTRY["execute_code"](self.current_code)
            self._last_error = exec_result["stderr"]
            if exec_result["timeout"]:
                self._timeout = True

        # search_error returns a suggestion — no state mutation needed

        # ---- done conditions ----
        # 1. Code passes all test assertions
        if tool_name in ("execute_code", "patch_code") and self._tests_pass():
            self.done = True

        # 2. Tool-call budget exhausted
        if len(self.tool_call_history) >= self.MAX_TOOL_CALLS:
            self.done = True

        reward = compute_reward(self.get_trajectory()) if self.done else 0.0
        return self._make_observation(), reward, self.done

    def get_trajectory(self) -> dict:
        """Return the trajectory dict expected by ``compute_reward``."""
        return {
            "final_code": self.current_code,
            "test_cases": self.sample["test_cases"],
            "tool_calls": [tc["tool"] for tc in self.tool_call_history],
            "timeout": self._timeout,
            "entry_point": self.sample["entry_point"],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _tests_pass(self) -> bool:
        """Run current_code together with the sample's test assertions."""
        entry = self.sample["entry_point"]
        tests = self.sample.get("test_cases", [])
        if not tests:
            return False

        # HumanEval tests reference ``candidate``; alias it to the entry point
        test_script = (
            self.current_code
            + f"\ncandidate = {entry}\n"
            + "\n".join(tests)
            + "\n"
        )
        result = TOOL_REGISTRY["execute_code"](test_script)
        return not result["stderr"] and not result["timeout"]

    def _make_observation(self) -> dict:
        return {
            "current_code": self.current_code,
            "error_msg": self._last_error,
            "tool_calls_used": len(self.tool_call_history),
            "done": self.done,
        }
