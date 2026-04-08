"""
FreshChain Post-Harvest Yield Loss Environment
inference.py  ←  MUST be in root directory (competition requirement)

This script runs an LLM agent against all 3 tasks and prints scores.
Uses OpenAI client as required by the competition.

Environment variables required:
  API_BASE_URL   - The API endpoint for the LLM
  MODEL_NAME     - The model identifier
  HF_TOKEN       - Your Hugging Face / API key
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

# Environment server URL (running locally via Docker or HF Space)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

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
    resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()

def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()

def env_grade() -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/grade")
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────
# AGENT SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an agricultural logistics AI agent managing a post-harvest cold-chain warehouse.

Your goal is to MAXIMIZE yield saved and MINIMIZE spoilage losses.

Each step you receive a JSON observation with:
- batches: list of produce batches (each has spoilage_risk, quantity_kg, crop_type, etc.)
- trucks: list of trucks (each has available, capacity_kg, truck_id)
- message: what happened last step
- done: whether the episode is over

You must respond with a JSON action ONLY. No explanation. No markdown. Just raw JSON.

Action format:
{
  "action_type": "dispatch" | "store" | "reroute" | "discard",
  "batch_id": "B001",      (required for dispatch/reroute/discard)
  "truck_id": "T01",       (required for dispatch)
  "destination": "Mumbai"  (required for reroute)
}

Strategy:
1. Prioritize dispatching batches with HIGH spoilage_risk (>0.5) first
2. Use available trucks for the highest-risk batches
3. If no trucks available, store and wait
4. Discard only if spoilage_risk > 0.85 to prevent cascade
5. Never waste trucks on low-risk batches when high-risk ones exist
"""

# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────

def run_task(task_id: str, max_steps: int = 20) -> float:
    print(f"\n{'='*50}", flush=True)
    print(f"  TASK: {task_id.upper()}", flush=True)
    print(f"{'='*50}", flush=True)

    obs = env_reset(task_id)
    print(f"  Initial batches: {len(obs.get('batches', []))}", flush=True)
    print(f"  Initial trucks:  {len(obs.get('trucks', []))}", flush=True)

    # ── REQUIRED: START block ──
    print(f"[START] task={task_id}", flush=True)

    current_step = 0
    for step in range(max_steps):
        if obs.get("done"):
            break

        current_step = step + 1

        # Ask LLM for action
        obs_str = json.dumps(obs, indent=2)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current observation:\n{obs_str}\n\nWhat action do you take?"}
                ]
            )
            action_str = response.choices[0].message.content.strip()

            # Clean up if LLM added markdown
            if action_str.startswith("```"):
                action_str = action_str.split("```")[1]
                if action_str.startswith("json"):
                    action_str = action_str[4:]

            action = json.loads(action_str)

        except json.JSONDecodeError:
            print(f"  Step {current_step}: LLM gave invalid JSON, defaulting to store", flush=True)
            action = {"action_type": "store"}
        except Exception as e:
            print(f"  Step {current_step}: LLM error ({e}), defaulting to store", flush=True)
            action = {"action_type": "store"}

        # Execute action
        result = env_step(action)
        obs = result.get("observation", result)
        reward = result.get("reward", 0.0)

        print(f"  Step {current_step}: {action.get('action_type', '?')} "
              f"| reward={reward:.3f} "
              f"| saved={obs.get('total_yield_saved_kg', 0):.0f}kg "
              f"| lost={obs.get('total_yield_lost_kg', 0):.0f}kg", flush=True)
        print(f"           {obs.get('message', '')[:80]}", flush=True)

        # ── REQUIRED: STEP block ──
        print(f"[STEP] step={current_step} reward={reward:.4f}", flush=True)

        if obs.get("done"):
            break

    # Get final grade
    grade_result = env_grade()
    score = grade_result.get("score", 0.0)
    saved = grade_result.get("yield_saved_kg", 0)
    lost = grade_result.get("yield_lost_kg", 0)

    print(f"\n  FINAL SCORE: {score:.3f}", flush=True)
    print(f"  Yield saved: {saved:.1f}kg | Yield lost: {lost:.1f}kg", flush=True)

    # ── REQUIRED: END block ──
    print(f"[END] task={task_id} score={score:.4f} steps={current_step}", flush=True)

    return score


# ─────────────────────────────────────────────
# MAIN — Run all 3 tasks
# ─────────────────────────────────────────────

def main():
    print("\nFreshChain Post-Harvest Yield Loss Environment", flush=True)
    print("Baseline Inference Script", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API:   {API_BASE_URL}", flush=True)

    # Wait for server to be ready
    print("\nWaiting for environment server...", flush=True)
    for _ in range(30):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server ready.", flush=True)
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("ERROR: Environment server not reachable at", ENV_BASE_URL, flush=True)
        return

    scores = {}
    task_configs = {
        "easy":   {"max_steps": 5},
        "medium": {"max_steps": 10},
        "hard":   {"max_steps": 15},
    }

    for task_id, cfg in task_configs.items():
        score = run_task(task_id, max_steps=cfg["max_steps"])
        scores[task_id] = score
        time.sleep(1)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    print("  BASELINE RESULTS SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:8s} | {score:.3f} | {bar}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"{'─'*50}", flush=True)
    print(f"  {'AVERAGE':8s} | {avg:.3f}", flush=True)
    print(f"{'='*50}\n", flush=True)


if __name__ == "__main__":
    main()
