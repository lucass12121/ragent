"""Gradio demo for the Code Debugging Agent.

Provides an interactive UI to paste buggy Python code and see
the step-by-step debugging process, fixed code, and reward score.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.debug_env import DebugEnv
from tools import TOOL_REGISTRY
from train.train import MockAgent, safe_compute_reward


# ---------------------------------------------------------------------------
# Core debugging function
# ---------------------------------------------------------------------------

def debug_code(buggy_code: str) -> tuple:
    """Debug buggy Python code using MockAgent.

    Args:
        buggy_code: Python source code with a bug.

    Returns:
        (debug_log, fixed_code, reward_score)
    """
    if not buggy_code or not buggy_code.strip():
        return "Please paste some Python code.", "", 0.0

    # Build a synthetic sample (no ground truth available in demo mode)
    sample = {
        "buggy_code": buggy_code,
        "correct_code": buggy_code,  # unknown in demo
        "test_cases": [],
        "entry_point": "unknown",
    }

    env = DebugEnv()
    agent = MockAgent(tool_names=list(TOOL_REGISTRY.keys()))

    observation = env.reset(sample)
    agent.reset()
    done = False

    log_lines = []
    step_num = 0

    # Record initial error
    initial_error = observation.get("error_msg", "")
    if initial_error:
        log_lines.append(f"Initial error: {initial_error.strip()}")
        log_lines.append("")

    while not done:
        step_num += 1
        action = agent.act(observation, env.current_code)
        tool_name = action["tool"]

        observation, reward, done = env.step(action)

        # Format step log
        error_msg = observation.get("error_msg", "").strip()
        if tool_name == "execute_code":
            if error_msg:
                result_str = f"Error: {error_msg[:200]}"
            else:
                result_str = "Execution successful"
        elif tool_name == "search_error":
            # Get suggestion from searcher
            from tools.searcher import search_error
            suggestion = search_error(action["args"].get("error_msg", ""))
            result_str = suggestion[:200]
        elif tool_name == "patch_code":
            result_str = "Code updated" if not error_msg else f"Patched, but error: {error_msg[:150]}"
        else:
            result_str = "Done"

        status = " [PASS]" if done and env._tests_pass() else ""
        log_lines.append(f"Step {step_num}: [{tool_name}] -> {result_str}{status}")

    # Compute reward
    trajectory = env.get_trajectory()
    final_reward = safe_compute_reward(trajectory)

    log_lines.append("")
    log_lines.append(f"Final reward: {final_reward:.2f}")

    debug_log = "\n".join(log_lines)
    fixed_code = env.current_code

    return debug_log, fixed_code, round(final_reward, 2)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_BUGGY_CODE = '''\
def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a - b  # bug: should be a + b
    return b
'''


def build_app():
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio is not installed. Run: pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="Code Debugging Agent Demo") as app:
        gr.Markdown("# 🔍 Code Debugging Agent Demo")
        gr.Markdown(
            "> **Note:** Currently using MockAgent for demonstration. "
            "Results will improve significantly after connecting a real trained model."
        )

        with gr.Row():
            with gr.Column():
                code_input = gr.Textbox(
                    label="Buggy Python Code",
                    placeholder="Paste your buggy Python code here...",
                    lines=15,
                    value=EXAMPLE_BUGGY_CODE,
                )
                debug_btn = gr.Button("Debug Code", variant="primary")

            with gr.Column():
                debug_log = gr.Textbox(
                    label="Debugging Process",
                    lines=15,
                    interactive=False,
                )

        with gr.Row():
            with gr.Column():
                fixed_code = gr.Textbox(
                    label="Fixed Code",
                    lines=10,
                    interactive=False,
                )
            with gr.Column():
                reward_score = gr.Number(
                    label="Reward Score",
                    interactive=False,
                )

        debug_btn.click(
            fn=debug_code,
            inputs=[code_input],
            outputs=[debug_log, fixed_code, reward_score],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = build_app()
    print("Starting Gradio demo ...")
    print("Note: Using MockAgent. Connect a real model for better results.")
    app.launch(share=False)


if __name__ == "__main__":
    main()
