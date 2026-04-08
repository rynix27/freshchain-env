"""
FreshChain Post-Harvest Yield Loss Environment
inference.py — Baseline Inference Script (OpenEnv Competition Requirement)

This script runs an LLM agent against all 3 tasks and prints scores.
Uses the OpenAI API client as required by the competition spec.

Environment variables:
  HF_TOKEN       - Your Hugging Face / API key (required)
  API_BASE_URL   - LLM API base URL (default: https://api.openai.com/v1)
  MODEL_NAME     - Model identifier (default: gpt-4o-mini)
  ENV_BASE_URL   - FreshChain server URL (default: http://localhost:7860)

Usage:
  export HF_TOKEN=your_token_here
  python inference.py

What this script does:
  1. Waits for the FreshChain environment server to be ready
  2. For each task (easy, medium, hard):
     a. Resets the environment to start a new episode
     b. Calls the LLM with the current observation as context
     c. Parses the LLM's JSON action response
     d. Steps the environment with that action
     e. Repeats until done=True
     f. Calls /grade to get the final task score
  3. Prints a summary table of all scores

Agent loop:
  observe -> LLM decides action -> act -> receive reward -> repeat
"""

import os
import json
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG FROM ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. Set it via: export HF_TOKEN=your_token")

# ─────────────────────────────────────────────
# OpenAI CLIENT
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
    base_url=API_BASE_URL,
)

# ─────────────────────────────────────────────
# ENVIRONMENT HELPER FUNCTIONS
# ─────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    """Reset the environment for a given task. Returns initial observation."""
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    """Take one step in the environment. Returns {observation, reward, done, info}."""
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_grade() -> dict:
    """Get the final score for the current episode. Returns {score, yield_saved_kg, ...}."""
    resp = requests.post(f"{ENV_BASE_URL}/grade", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an agricultural logistics AI agent managing a post-harvest cold-chain warehouse in India.

GOAL: Maximize produce saved (dispatched to market) and minimize spoilage losses.

OBSERVATION (JSON you receive each step):
- batches: list of produce batches, each with:
    - batch_id: e.g. "B001"
    - crop_type: e.g. "tomato"
    - quantity_kg: amount remaining
    - spoilage_risk: 0.0 (fresh) to 1.0 (fully spoiled) — KEY SIGNAL
    - temperature_c: high temp = faster spoilage
    - market_price_per_kg: current APMC price in INR
- trucks: list of trucks, each with:
    - truck_id: e.g. "T01"
    - available: true/false (trucks may break down!)
    - capacity_kg: max load
- done: true = episode is over
- reward: what you earned last step

ACTIONS you can take (respond with JSON ONLY, no markdown, no explanation):
{
  "action_type": "dispatch",  // sends batch to market
  "batch_id": "B001",
  "truck_id": "T01"
}
OR:
{"action_type": "store"}  // wait one step (spoilage increases!)
OR:
{"action_type": "reroute", "batch_id": "B001", "destination": "Vashi APMC (Mumbai)"}
OR:
{"action_type": "discard", "batch_id": "B001"}  // only if spoilage_risk > 0.85

STRATEGY:
1. Always dispatch highest spoilage_risk batches first (risk > 0.5 = urgent!)
2. Match high-risk batches with available trucks immediately
3. If no trucks available, store and wait — don't discard healthy batches
4. Discard only if spoilage_risk > 0.85 AND no truck available (prevents cascade)
5. Reroute if a batch's destination has low price and a better market exists
6. Never waste a truck on a low-risk batch when a high-risk one exists

RESPOND WITH ONLY A JSON OBJECT. No text before or after."""


# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────

def run_task(task_id: str, max_steps: int = 20) -> float:
    """
    Run one full episode of the given task.
    
    Loop:
      1. Reset environment
      2. Observe current state
      3. LLM picks action
      4. Execute action, receive reward
      5. Repeat until done
      6. Grade the episode
    
    Returns final score (0.0 to 1.0).
    """
    print(f"\n{'='*55}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*55}")

    obs = env_reset(task_id)
    print(f"  Batches in storage: {len(obs.get('batches', []))}")
    print(f"  Trucks available:   {len([t for t in obs.get('trucks', []) if t.get('available')])}")

    total_reward = 0.0

    for step_num in range(max_steps):
        if obs.get("done"):
            break

        # Format observation for LLM
        obs_str = json.dumps(obs, indent=2)

        # ── Ask LLM for action ──
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=300,
                temperature=0.1,  # Low temperature for deterministic, strategic choices
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Step {step_num + 1} observation:\n{obs_str}\n\n"
                            f"What action do you take? Respond with JSON only."
                        )
                    }
                ]
            )
            action_str = response.choices[0].message.content.strip()

            # Strip markdown code fences if LLM added them
            if action_str.startswith("```"):
                lines = action_str.split("\n")
                action_str = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )

            action = json.loads(action_str.strip())

        except json.JSONDecodeError:
            print(f"  Step {step_num+1}: LLM gave invalid JSON, defaulting to store")
            action = {"action_type": "store"}
        except Exception as e:
            print(f"  Step {step_num+1}: LLM error ({type(e).__name__}: {e}), defaulting to store")
            action = {"action_type": "store"}

        # ── Execute action in environment ──
        result = env_step(action)
        obs = result.get("observation", result)
        reward = result.get("reward", 0.0)
        total_reward += reward

        # ── Print step summary ──
        saved = obs.get("total_yield_saved_kg", 0)
        lost = obs.get("total_yield_lost_kg", 0)
        msg = obs.get("message", "")[:80]
        print(
            f"  Step {step_num+1:2d}: [{action.get('action_type', '?'):8s}] "
            f"reward={reward:+.3f} | saved={saved:.0f}kg | lost={lost:.0f}kg"
        )
        if msg:
            print(f"           {msg}")

        if obs.get("done"):
            break

    # ── Get final grade ──
    grade_result = env_grade()
    score = grade_result.get("score", 0.0)
    saved = grade_result.get("yield_saved_kg", 0)
    lost = grade_result.get("yield_lost_kg", 0)
    steps = grade_result.get("steps_used", 0)

    print(f"\n  ── FINAL RESULTS ──")
    print(f"  Score:       {score:.3f} / 1.000")
    print(f"  Yield saved: {saved:.1f} kg")
    print(f"  Yield lost:  {lost:.1f} kg")
    print(f"  Steps used:  {steps}")
    print(f"  Total reward (episode): {total_reward:.3f}")

    return score


# ─────────────────────────────────────────────
# WAIT FOR SERVER
# ─────────────────────────────────────────────

def wait_for_server(timeout_seconds: int = 60):
    """Poll /health until server is ready or timeout."""
    print(f"\nWaiting for environment server at {ENV_BASE_URL} ...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"ERROR: Environment server not reachable at {ENV_BASE_URL}")
    return False


# ─────────────────────────────────────────────
# MAIN — Run all 3 tasks
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  FreshChain Post-Harvest Yield Loss Environment")
    print("  Baseline Inference Script")
    print("="*55)
    print(f"  Model:   {MODEL_NAME}")
    print(f"  API:     {API_BASE_URL}")
    print(f"  Server:  {ENV_BASE_URL}")

    if not wait_for_server():
        return

    task_configs = {
        "easy":   {"max_steps": 5},
        "medium": {"max_steps": 10},
        "hard":   {"max_steps": 15},
    }

    scores = {}
    for task_id, cfg in task_configs.items():
        score = run_task(task_id, max_steps=cfg["max_steps"])
        scores[task_id] = score
        time.sleep(1)

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Task':8s} | {'Score':6s} | {'Bar (0→1)':22s} | {'Pass?':6s}")
    print(f"  {'-'*50}")

    thresholds = {"easy": 0.80, "medium": 0.60, "hard": 0.40}
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        threshold = thresholds[task_id]
        status = " PASS" if score >= threshold else " FAIL"
        print(f"  {task_id:8s} | {score:.3f} | {bar} | {status}")

    avg = sum(scores.values()) / len(scores)
    print(f"  {'-'*50}")
    print(f"  {'AVERAGE':8s} | {avg:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
