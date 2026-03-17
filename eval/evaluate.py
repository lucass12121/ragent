"""Evaluation script for the Code Debugging Agent.

Compares baseline (MockAgent) vs trained agent on DebugDataset,
computes metrics, and optionally plots training curves.
"""

import argparse
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DebugDataset
from env.debug_env import DebugEnv
from reward.reward_fn import compute_reward
from tools import TOOL_REGISTRY
from train.train import MockAgent, safe_compute_reward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(agent, dataset, num_samples: int = 50) -> dict:
    """Run agent on num_samples from dataset, return evaluation metrics.

    Returns:
        {
            "success_rate": float,   # fraction of episodes where all tests passed
            "avg_reward": float,     # mean reward across episodes
            "avg_tool_calls": float, # mean number of tool calls per episode
            "pass_at_1": float,      # fraction where first patch action succeeded
            "timeout_rate": float,   # fraction of episodes that timed out
        }
    """
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    env = DebugEnv()
    successes = 0
    rewards = []
    tool_counts = []
    pass_at_1_count = 0
    timeout_count = 0

    for idx in indices:
        sample = dataset[idx]
        observation = env.reset(sample)
        agent.reset()
        done = False

        first_patch_done = False
        first_patch_success = False

        while not done:
            action = agent.act(observation, env.current_code)
            observation, reward, done = env.step(action)

            # Track first patch_code result
            if action["tool"] == "patch_code" and not first_patch_done:
                first_patch_done = True
                if done and env._tests_pass():
                    first_patch_success = True

        trajectory = env.get_trajectory()
        ep_reward = safe_compute_reward(trajectory)

        # Check success: tests pass on final code
        success = env._tests_pass()
        successes += int(success)
        rewards.append(ep_reward)
        tool_counts.append(len(trajectory["tool_calls"]))
        if first_patch_success:
            pass_at_1_count += 1
        if trajectory["timeout"]:
            timeout_count += 1

    return {
        "success_rate": successes / num_samples,
        "avg_reward": sum(rewards) / num_samples,
        "avg_tool_calls": sum(tool_counts) / num_samples,
        "pass_at_1": pass_at_1_count / num_samples,
        "timeout_rate": timeout_count / num_samples,
    }


def compare_agents(baseline_agent, trained_agent, dataset, num_samples: int = 50) -> dict:
    """Compare two agents and return metrics with improvement.

    Returns:
        {
            "baseline": dict,     # baseline metrics
            "trained": dict,      # trained agent metrics
            "improvement": dict,  # percentage improvement per metric
        }
    """
    print("Evaluating baseline agent ...")
    baseline_results = run_evaluation(baseline_agent, dataset, num_samples)

    print("Evaluating trained agent ...")
    trained_results = run_evaluation(trained_agent, dataset, num_samples)

    improvement = {}
    for key in baseline_results:
        base_val = baseline_results[key]
        train_val = trained_results[key]
        if base_val != 0:
            improvement[key] = ((train_val - base_val) / abs(base_val)) * 100
        else:
            improvement[key] = float("inf") if train_val > 0 else 0.0

    return {
        "baseline": baseline_results,
        "trained": trained_results,
        "improvement": improvement,
    }


def save_results(results: dict, path: str = "eval/results.json"):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {path}")


def plot_training_curve(log_path: str = "logs/training.log"):
    """Read training log and plot reward curve to eval/reward_curve.png.

    Expected log format (one per line):
        [iter 1/100] mean_reward=0.1234  rewards=[0.1, 0.2, ...]
    """
    import re

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    iterations = []
    mean_rewards = []
    pattern = re.compile(r"\[iter\s+(\d+)/\d+\]\s+mean_reward=([\d.\-]+)")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iterations.append(int(m.group(1)))
                mean_rewards.append(float(m.group(2)))

    if not iterations:
        print("No training data found in log.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_rewards, "b-", linewidth=1.5, label="Mean Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = "eval/reward_curve.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reward curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_metrics(label: str, metrics: dict):
    parts = [f"{k}={v:.2f}" for k, v in metrics.items()]
    print(f"[{label}]  {'  '.join(parts)}")


def print_improvement(improvement: dict):
    parts = []
    for k, v in improvement.items():
        sign = "+" if v >= 0 else ""
        parts.append(f"{k} {sign}{v:.0f}%")
    print(f"Improvement: {'  '.join(parts)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Debugging Agent")
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of samples to evaluate on",
    )
    parser.add_argument(
        "--output", type=str, default="eval/results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot training reward curve from logs/training.log",
    )
    args = parser.parse_args()

    # Load test dataset
    print("Loading test dataset ...")
    dataset = DebugDataset(split="test")
    print(f"Dataset loaded: {len(dataset)} samples")

    tool_names = list(TOOL_REGISTRY.keys())

    # Baseline: standard MockAgent (random actions)
    baseline_agent = MockAgent(tool_names=tool_names)

    # "Trained" agent: also MockAgent for now (placeholder for real model)
    # When GPU is available, replace with the real trained agent
    trained_agent = MockAgent(tool_names=tool_names)

    # Run comparison
    results = compare_agents(
        baseline_agent, trained_agent, dataset,
        num_samples=args.samples,
    )

    # Print results
    print()
    print_metrics("Baseline", results["baseline"])
    print_metrics("Trained ", results["trained"])
    print_improvement(results["improvement"])

    # Save
    save_results(results, args.output)

    # Optional plot
    if args.plot:
        plot_training_curve()


if __name__ == "__main__":
    main()
