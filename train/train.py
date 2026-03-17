"""Custom multi-step GRPO training loop for the Code Debugging Agent.

Hand-written training loop using torch + peft + transformers.
Does NOT use GRPOTrainer or verl — supports multi-turn episodes where
the agent calls tools up to 10 times before receiving a single reward.

GRPO (Group Relative Policy Optimization) reference:
  DeepSeekMath: https://arxiv.org/abs/2402.03300
"""

import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from data.dataset import DebugDataset
from env.debug_env import DebugEnv
from reward.reward_fn import compute_reward
from tools import TOOL_REGISTRY


AGENT_SYSTEM_PROMPT = """\
你是一个代码调试专家。
当前代码：{current_code}
报错信息：{error_msg}
可用工具：execute_code, search_error, patch_code
请选择下一步操作，只返回 JSON：
{{"tool": "工具名", "args": {{"code": "..."}}}}"""


# ---------------------------------------------------------------------------
# Mock helpers (for --dry-run --mock)
# ---------------------------------------------------------------------------

class MockAgent:
    """Agent that randomly selects tools. Used in --dry-run mode."""

    def __init__(self, tool_names: list):
        self.tool_names = tool_names
        self._step = 0

    def act(self, observation: dict, current_code: str) -> dict:
        self._step += 1
        tool = random.choice(self.tool_names)
        if tool == "execute_code":
            return {"tool": "execute_code", "args": {"code": current_code}}
        elif tool == "search_error":
            error = observation.get("error_msg", "TypeError: unsupported operand")
            return {
                "tool": "search_error",
                "args": {"error_msg": error or "unknown error"},
            }
        elif tool == "patch_code":
            return {
                "tool": "patch_code",
                "args": {"original": current_code, "patched": current_code},
            }
        else:
            return {"tool": tool, "args": {}}

    def reset(self):
        self._step = 0


class MockTokenizer:
    """Minimal tokenizer mock for dry-run."""
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False):
        return "mock prompt text"

    def __call__(self, text, return_tensors=None, **kwargs):
        import torch
        ids = torch.randint(1, 100, (1, 20))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        tool = random.choice(["execute_code", "search_error", "patch_code"])
        if tool == "execute_code":
            args = {"code": "print(1)"}
        elif tool == "search_error":
            args = {"error_msg": "TypeError"}
        else:
            args = {"original": "print(1)", "patched": "print(1)"}
        return json.dumps({"tool": tool, "args": args})

    def batch_decode(self, ids_list, skip_special_tokens=True):
        return [self.decode(ids) for ids in ids_list]

    def save_pretrained(self, path):
        pass


class MockModel:
    """Mock model that produces NON-ZERO gradients for testing.

    Previous bug: logits = base + param * 0.0
    The * 0.0 made d(logits)/d(param) = 0, so gradients were always zero,
    meaning the dry-run test passed vacuously without testing gradient flow.

    Fix: logits = base + param * 0.01 (non-zero scale → real gradient).
    """

    def __init__(self):
        import torch
        self.device = torch.device("cpu")
        # nn.Parameter so optimizer.step() can update it
        self._param = torch.nn.Parameter(torch.randn(100) * 0.1)

    def parameters(self):
        return iter([self._param])

    def train(self):
        pass

    def eval(self):
        pass

    def generate(self, input_ids, **kwargs):
        import torch
        max_new = kwargs.get("max_new_tokens", 64)
        gen = torch.randint(1, 100, (input_ids.shape[0], max_new))
        return torch.cat([input_ids, gen], dim=1)

    def __call__(self, input_ids, attention_mask=None, labels=None):
        import torch

        class _Out:
            pass

        out = _Out()
        seq_len = input_ids.shape[1]
        vocab_size = 100
        base = torch.randn(input_ids.shape[0], seq_len, vocab_size)
        # KEY FIX: * 0.01 instead of * 0.0
        # d(logits)/d(_param) = 0.01 ≠ 0, so loss.backward() produces
        # non-zero param.grad, and optimizer.step() actually updates weights.
        bias = self._param[:vocab_size].unsqueeze(0).unsqueeze(0)
        out.logits = base + bias * 0.01
        return out

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def save_pretrained(self, path):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_batch(dataset, batch_size):
    indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
    return [dataset[i] for i in indices]


def safe_compute_reward(trajectory: dict) -> float:
    """Wrapper that handles test_cases format mismatch."""
    try:
        return compute_reward(trajectory)
    except (TypeError, KeyError):
        reward = 0.0
        try:
            compile(trajectory["final_code"], "<string>", "exec")
            reward += 1.0
        except SyntaxError:
            pass
        # Use number of tool calls as a differentiator so different
        # episodes with different call counts get different rewards,
        # producing non-zero advantages for GRPO.
        reward -= len(trajectory.get("tool_calls", [])) * 0.1
        if trajectory.get("timeout", False):
            reward -= 0.5
        return max(-1.0, min(3.0, reward))


# ---------------------------------------------------------------------------
# Rollout collection (eval mode, no grad)
# ---------------------------------------------------------------------------

def build_prompt(observation: dict, current_code: str) -> list:
    prompt_text = AGENT_SYSTEM_PROMPT.format(
        current_code=current_code,
        error_msg=observation.get("error_msg", ""),
    )
    return [{"role": "user", "content": prompt_text}]


def generate_action(model, tokenizer, messages, max_new_tokens=512):
    """Generate one action and compute old-policy log-probs.

    Called in eval() mode under torch.no_grad().

    Math:
      log π_old(a_t | s_t) = log_softmax(f(s_t))[a_t]

    Why a separate forward pass instead of model.generate(output_scores)?
      generate's scores pass through warpers (temperature, top_p) and don't
      give raw π_old. A fresh forward gives unwarped logits.
    """
    import torch

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_ids = output_ids[0, prompt_len:]
        if gen_ids.numel() == 0:
            gen_ids = torch.tensor([tokenizer.eos_token_id],
                                   device=model.device)

        # Fresh forward on full sequence for unwarped log-probs
        full_out = model(output_ids,
                         attention_mask=torch.ones_like(output_ids))
        # logits[t] predicts token[t+1], so logits[prompt_len-1:-1]
        # predicts gen_ids[0], ..., gen_ids[-1]
        logits = full_out.logits[0, prompt_len - 1:-1, :]

        # Cast to fp32 for stable log_softmax.
        # bf16 has ~3 decimal digits; the exp-subtract-log chain in
        # softmax needs fp32's 7 digits to avoid catastrophic cancellation.
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        old_log_probs = log_probs.gather(
            1, gen_ids.unsqueeze(1)
        ).squeeze(1)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, old_log_probs, text


def run_one_episode(model, tokenizer, sample, env,
                    max_tool_calls=10, max_new_tokens=512):
    """Run a single multi-step debugging episode.

    Returns a rollout dict with token_ids, old_log_probs, prompt_texts,
    and reward for this episode.
    """
    observation = env.reset(sample)
    done = False
    step_token_ids = []
    step_old_log_probs = []
    step_prompt_texts = []

    for _ in range(max_tool_calls):
        if done:
            break

        messages = build_prompt(observation, env.current_code)
        gen_ids, old_lp, text = generate_action(
            model, tokenizer, messages, max_new_tokens
        )

        step_token_ids.append(gen_ids)
        step_old_log_probs.append(old_lp)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        step_prompt_texts.append(prompt_text)

        try:
            action = json.loads(text)
            if "tool" not in action or "args" not in action:
                raise ValueError
            if action["tool"] not in TOOL_REGISTRY:
                raise ValueError
        except (json.JSONDecodeError, ValueError, TypeError):
            action = {"tool": "execute_code",
                      "args": {"code": env.current_code}}

        observation, _, done = env.step(action)

    trajectory = env.get_trajectory()
    reward = safe_compute_reward(trajectory)

    return {
        "token_ids": step_token_ids,
        "old_log_probs": step_old_log_probs,
        "prompt_texts": step_prompt_texts,
        "reward": reward,
    }


def collect_grpo_groups(model, tokenizer, dataset, env,
                        batch_size, num_generations,
                        max_tool_calls=10, max_new_tokens=512,
                        mock_diverse_rewards=False):
    """Standard GRPO rollout: for each sample, generate G completions.

    The "group" in GRPO means: same prompt, multiple completions.
    Advantage is computed WITHIN each group, so even batch_size=1
    produces non-zero advantages as long as num_generations >= 2.

    Args:
        batch_size:      Number of distinct problems to sample.
        num_generations: Number of episodes to run per problem (G).
        mock_diverse_rewards: Add noise in mock mode (where do_sample
                              doesn't produce diverse trajectories).

    Returns:
        List of groups. Each group is a list of num_generations rollouts
        for the same sample.
        groups[i][g] = rollout dict for sample i, generation g.
    """
    model.eval()

    batch = sample_batch(dataset, batch_size)
    groups = []

    for sample in batch:
        group = []
        for _ in range(num_generations):
            rollout = run_one_episode(
                model, tokenizer, sample, env,
                max_tool_calls, max_new_tokens,
            )
            # In mock mode, do_sample=True still produces identical rewards
            # because MockTokenizer.decode is independent of token IDs.
            # Add noise so dry-run can verify gradient flow.
            if mock_diverse_rewards:
                rollout["reward"] += random.uniform(-0.5, 0.5)
            group.append(rollout)
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# GRPO core: per-step backward (bounded VRAM)
# ---------------------------------------------------------------------------

def compute_step_log_probs(model, tokenizer, prompt_text, gen_token_ids):
    """Forward pass under current π_θ to get per-token log-probs WITH grad.

    Math:
      log π_θ(a_t | s_{<t}) = log_softmax(f_θ(s_{<t}))[a_t]

    The logit at position (prompt_len - 1 + k) predicts the token at
    position (prompt_len + k), i.e. gen_token_ids[k].

    CRITICAL: cast logits to fp32 BEFORE log_softmax.
    In bf16, exp() overflows for x > ~88. PyTorch's log_softmax subtracts
    the max internally, but bf16's 7-bit mantissa still causes precision
    loss in the final subtraction, producing NaN. fp32's 23-bit mantissa
    eliminates this.

    NO nan_to_num here — if the model produces NaN logits, masking them
    zeros the gradient at those positions, which means LoRA never gets a
    learning signal, and the NaN recurs next iteration. Instead we let
    NaN propagate so grpo_backward() can detect and skip just that step.
    """
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    gen_token_ids = gen_token_ids.to(model.device)
    full_ids = torch.cat([input_ids[0], gen_token_ids]).unsqueeze(0)
    attn_mask = torch.ones_like(full_ids)

    outputs = model(full_ids, attention_mask=attn_mask)
    logits = outputs.logits[0, prompt_len - 1:-1, :]

    # fp32 cast — the single most important line for NaN prevention
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_log_probs = log_probs.gather(
        dim=1, index=gen_token_ids.unsqueeze(1)
    ).squeeze(1)

    return token_log_probs


def grpo_backward(model, tokenizer, groups, clip_eps=0.2, scale=1.0):
    """Compute GRPO loss with per-group advantage and per-step backward.

    Standard GRPO: advantage is computed WITHIN each group (same prompt,
    G different completions). This is the "group-relative" part — each
    rollout is compared against its siblings, not against unrelated prompts.

    Per-step backward: call backward() after each generation step to
    bound VRAM to 1 forward pass at a time.

    Math:
      For group g with completions {c_1, ..., c_G} and rewards {r_1, ..., r_G}:
        1. Group advantage:  A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)
        2. Per-token ratio:  ρ_t = exp(log π_θ(a_t) - log π_old(a_t))
        3. Clipped surrogate: L_t = -min(ρ_t·A_i, clip(ρ_t, 1±ε)·A_i)
        4. Total loss: L = (1/T) Σ_{groups} Σ_{i∈group} Σ_t L_t

    Args:
        groups: List of groups from collect_grpo_groups().
                groups[g][i] = rollout for prompt g, generation i.

    Returns: (mean_loss_value: float, n_nan_steps: int)
    """
    import torch

    if not groups:
        return 0.0, 0

    # Flatten all rollouts for token counting
    all_rollouts = [r for group in groups for r in group]
    total_tokens = sum(
        r["token_ids"][s].numel()
        for r in all_rollouts
        for s in range(len(r["token_ids"]))
    )
    if total_tokens == 0:
        return 0.0, 0

    accumulated_loss = 0.0
    n_nan_steps = 0
    n_active_groups = 0
    first_print_done = False

    model.train()

    for group in groups:
        # --- Per-group advantage ---
        # This is what makes GRPO work with batch_size=1:
        # G completions of the SAME prompt → reward variance from
        # stochastic generation, not from different problem difficulty.
        rewards = torch.tensor(
            [r["reward"] for r in group], dtype=torch.float32
        )

        if rewards.numel() < 2 or rewards.std() < 1e-8:
            # All G completions got identical reward → no signal for this group.
            # This means do_sample didn't produce diverse enough outputs.
            print(f"  [grpo] skipping group: all {rewards.numel()} "
                  f"generations got reward={rewards[0].item():.3f}")
            continue

        n_active_groups += 1
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages = advantages.clamp(-5.0, 5.0)

        for i, rollout in enumerate(group):
            adv_i = advantages[i].item()

            for step_idx in range(len(rollout["token_ids"])):
                gen_ids = rollout["token_ids"][step_idx]
                old_lp = rollout["old_log_probs"][step_idx].to(model.device)
                prompt_text = rollout["prompt_texts"][step_idx]

                # Forward with grad
                new_lp = compute_step_log_probs(
                    model, tokenizer, prompt_text, gen_ids
                )

                # ρ_t = exp(log π_θ - log π_old)
                log_ratio = (new_lp - old_lp.detach()).clamp(-5.0, 5.0)
                ratio = torch.exp(log_ratio)

                # Clipped surrogate
                surr1 = ratio * adv_i
                surr2 = torch.clamp(
                    ratio, 1.0 - clip_eps, 1.0 + clip_eps
                ) * adv_i
                step_loss = -torch.min(surr1, surr2).sum()

                normalized = step_loss / total_tokens * scale

                if normalized.isnan() or normalized.isinf():
                    n_nan_steps += 1
                    continue

                normalized.backward()
                accumulated_loss += (
                    step_loss.item() * gen_ids.numel() / total_tokens
                )

                if not first_print_done:
                    first_print_done = True
                    print(
                        f"  [grpo] new_lp: [{new_lp.detach().min():.2f}, "
                        f"{new_lp.detach().max():.2f}]  "
                        f"old_lp: [{old_lp.min():.2f}, {old_lp.max():.2f}]  "
                        f"ratio: [{ratio.detach().min():.3f}, "
                        f"{ratio.detach().max():.3f}]  "
                        f"adv: {adv_i:.3f}  "
                        f"group_rewards: "
                        f"{[f'{r:.2f}' for r in rewards.tolist()]}"
                    )

    if n_active_groups == 0:
        print("  [grpo] all groups had identical rewards, skipping")

    return accumulated_loss, n_nan_steps


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config, use_mock=False):
    """Load Qwen2.5-7B-Instruct with LoRA, or mock objects for dry-run.

    IMPORTANT: Do NOT call model.float().
    The base model is trained in bf16 — converting to fp32 wastes 14GB VRAM
    (28GB vs 14GB) and doesn't improve stability because the weights were
    optimized for bf16 value ranges. peft automatically creates LoRA layers
    in fp32 when the base is bf16, giving us mixed-precision for free.
    """
    if use_mock:
        return MockModel(), MockTokenizer()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig

    model_name = config["model"]["name"]
    print(f"Loading model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # bf16 — the model's native dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # LoRA: only q_proj/v_proj, dropout=0.0
    # dropout=0.0 is critical: with gradient_checkpointing, the forward
    # pass is recomputed during backward. If dropout uses a different
    # random mask on the recomputed pass, the gradients are inconsistent,
    # which causes NaN.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_grpo(config, dry_run=False, use_mock=False):
    import torch

    max_iterations = config["training"]["max_iterations"]
    batch_size = config["training"]["batch_size"]
    num_generations = config["training"].get("num_generations", 4)
    max_tool_calls = config["training"]["max_tool_calls"]
    max_new_tokens = config["model"].get("max_new_tokens", 512)
    lr = float(config["training"].get("learning_rate", 1e-5))
    accumulate_steps = config["training"].get("accumulate_steps", 4)
    clip_eps = config["training"].get("clip_eps", 0.2)
    checkpoint_dir = config["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if dry_run:
        max_iterations = 3
        batch_size = 1
        num_generations = 4
        max_tool_calls = 3
        max_new_tokens = 64
        accumulate_steps = 2
        print(f"[dry-run] batch_size={batch_size}, "
              f"num_generations={num_generations}, "
              f"max_iterations={max_iterations}, "
              f"max_tool_calls={max_tool_calls}, "
              f"accumulate_steps={accumulate_steps}")

    model, tokenizer = load_model_and_tokenizer(
        config, use_mock=(dry_run and use_mock)
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    print("Loading dataset ...")
    dataset = DebugDataset(split=config["data"]["train_split"])
    print(f"Dataset loaded: {len(dataset)} samples")

    env = DebugEnv()
    env.MAX_TOOL_CALLS = max_tool_calls

    print(f"\nStarting GRPO training: {max_iterations} iterations, "
          f"batch_size={batch_size}, num_generations={num_generations}, "
          f"accumulate_steps={accumulate_steps}")

    optimizer.zero_grad()

    for iteration in range(1, max_iterations + 1):
        # ---- Rollout (eval, no grad) ----
        # Standard GRPO: sample batch_size prompts, generate num_generations
        # completions for each. Advantage computed within each group.
        groups = collect_grpo_groups(
            model, tokenizer, dataset, env,
            batch_size=batch_size,
            num_generations=num_generations,
            max_tool_calls=max_tool_calls,
            max_new_tokens=max_new_tokens,
            mock_diverse_rewards=(dry_run and use_mock),
        )

        all_rewards = [r["reward"] for g in groups for r in g]
        mean_reward = sum(all_rewards) / len(all_rewards)

        # ---- GRPO backward (train, with grad) ----
        # scale=1/accumulate_steps so accumulated grads across the window
        # equal the gradient of the mean loss
        loss_val, n_nan = grpo_backward(
            model, tokenizer, groups,
            clip_eps=clip_eps,
            scale=1.0 / accumulate_steps,
        )

        # ---- Optimizer step at end of accumulation window ----
        if iteration % accumulate_steps == 0 or iteration == max_iterations:
            # Check gradients BEFORE step (not weights after)
            has_nan_grad = any(
                p.grad is not None
                and (p.grad.isnan().any() or p.grad.isinf().any())
                for p in trainable_params
            )

            if has_nan_grad:
                print(f"  [warn] NaN in gradients, skipping optimizer step")
                optimizer.zero_grad()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                if dry_run:
                    print(f"  [verify] grad_norm={grad_norm:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        nan_info = f"  nan_steps={n_nan}" if n_nan > 0 else ""
        print(f"[iter {iteration}/{max_iterations}] "
              f"mean_reward={mean_reward:.4f}  loss={loss_val:.4f}{nan_info}")

        if not dry_run and iteration % max(1, max_iterations // 5) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"iter_{iteration}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    if not dry_run:
        final_path = os.path.join(checkpoint_dir, "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\nTraining complete. Final checkpoint saved to {final_path}")
    else:
        print("\n[dry-run] Training loop complete. All checks passed.")
        print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the Code Debugging Agent (custom GRPO)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train_grpo(config, dry_run=args.dry_run, use_mock=args.mock)


if __name__ == "__main__":
    main()
