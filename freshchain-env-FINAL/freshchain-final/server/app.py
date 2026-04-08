"""
FreshChain Post-Harvest Yield Loss Environment
server/app.py — FastAPI server exposing the OpenEnv interface

Endpoints (OpenEnv required):
  POST /reset       → Start new episode, returns initial observation
  POST /step        → Agent takes one action, returns (observation, reward, done, info)
  GET  /state       → Returns current episode state

Extra endpoints:
  GET  /health      → Health check
  GET  /web         → Live warehouse dashboard UI
  GET  /tasks       → List all tasks with descriptions
  POST /grade       → Final score 0.0-1.0 for current episode
  GET  /alerts      → WhatsApp alert log for current episode
  WS   /ws          → WebSocket interface for real-time agent interaction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import json

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

from models import FreshChainAction
from server.environment import FreshChainEnvironment
from server.whatsapp_alerts import alert_system

app = FastAPI(
    title="FreshChain Post-Harvest Yield Loss Environment",
    description=(
        "An OpenEnv-compatible RL environment where an AI agent manages "
        "post-harvest produce logistics to minimize yield loss from spoilage. "
        "Based on real Indian APMC market data."
    ),
    version="1.0.0",
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# One environment instance per server
env = FreshChainEnvironment()


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
def dashboard():
    """Live warehouse dashboard UI."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ─────────────────────────────────────────────
# CORE OpenEnv ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "FreshChain Post-Harvest Yield Loss Environment",
        "version": "1.0.0",
        "description": (
            "An RL environment where an agent observes a cold-chain warehouse "
            "and takes actions (dispatch, store, reroute, discard) to minimize "
            "post-harvest produce spoilage in Indian agricultural supply chains."
        ),
        "tasks": ["easy", "medium", "hard"],
        "observation": "batches, trucks, spoilage_risk, yield_saved, yield_lost, reward, done",
        "actions": ["dispatch", "store", "reroute", "discard"],
        "reward": "positive for saving yield, negative for spoilage and poor decisions",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health", "/web", "/tasks", "/alerts"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = "easy"):
    """
    [OpenEnv] Start a new episode.

    task_id options:
      - "easy"   → 1 batch, 1 truck, basic spoilage control
      - "medium" → 3 batches, 2 trucks, truck breakdown at step 3
      - "hard"   → 6 batches, 2 trucks, breakdown + price volatility

    Returns: initial FreshChainObservation
    """
    obs = env.reset(task_id=task_id)
    return obs.model_dump()


@app.post("/step")
def step(action: FreshChainAction):
    """
    [OpenEnv] Agent takes one action. Returns observation, reward, done, info.

    Action format:
      {
        "action_type": "dispatch" | "store" | "reroute" | "discard",
        "batch_id": "B001",       (required for dispatch/reroute/discard)
        "truck_id": "T01",        (required for dispatch)
        "destination": "Mumbai"   (required for reroute)
      }

    Returns:
      {
        "observation": {...},
        "reward": float,
        "done": bool,
        "info": {"task_score": float, "step": int}
      }
    """
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "info": {
            "task_score": obs.task_score,
            "step": obs.step,
            "message": obs.message,
        }
    }


@app.get("/state")
def state():
    """[OpenEnv] Returns current episode metadata (episode_id, step_count, task_id, max_steps)."""
    return env.state.model_dump()


@app.post("/grade")
def grade():
    """
    Return the final task score for the current episode (0.0 to 1.0).

    Score formula:
      base_score        = yield_saved / (yield_saved + yield_lost)
      efficiency_bonus  = max(0, (max_steps - steps_used) / max_steps) * 0.1
      final_score       = min(1.0, base_score + efficiency_bonus)

    Grading criteria per task:
      easy   → score >= 0.8 = success (dispatch the single batch promptly)
      medium → score >= 0.6 = success (handle breakdown, prioritize high-risk)
      hard   → score >= 0.4 = success (manage volatility and cascade spoilage)
    """
    score = env.grade()
    return {
        "score": score,
        "yield_saved_kg": round(env._total_saved, 1),
        "yield_lost_kg": round(env._total_lost, 1),
        "episode_id": env.state.episode_id,
        "task_id": env.state.task_id,
        "steps_used": env.state.step_count,
        "max_steps": env.state.max_steps,
        "grading_note": (
            "Score = saved/(saved+lost) + efficiency_bonus. "
            "Bonus for finishing early. Penalized for spoilage cascade."
        ),
    }


# ─────────────────────────────────────────────
# INFORMATIONAL ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List all tasks with descriptions, difficulty, and expected scores."""
    from server.environment import TASK_CONFIGS
    task_info = {
        "easy": {
            "expected_agent_score": "0.80–1.00",
            "grader_threshold": 0.8,
            "objective": "Dispatch the single batch before spoilage_risk reaches 1.0",
            "scoring": "score = saved_kg / (saved_kg + lost_kg) + early_finish_bonus",
        },
        "medium": {
            "expected_agent_score": "0.55–0.75",
            "grader_threshold": 0.6,
            "objective": "Manage 3 batches with truck breakdown at step 3; prioritize highest-risk",
            "scoring": "Same formula. Bonus for saving all 3. Penalty if truck was idle on low-risk.",
        },
        "hard": {
            "expected_agent_score": "0.35–0.55",
            "grader_threshold": 0.4,
            "objective": "Handle 6 batches with breakdown + price volatility + spoilage cascade",
            "scoring": "Same formula. Rerouting to better markets yields extra reward.",
        },
    }
    return {
        task_id: {
            **cfg,
            **task_info.get(task_id, {}),
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


@app.get("/alerts")
def get_alerts():
    """Return all WhatsApp alerts generated during the current episode."""
    return {"alerts": alert_system.get_alerts()}


@app.get("/alerts/recent")
def get_recent_alerts(n: int = 5):
    """Return the most recent n alerts."""
    return {"alerts": alert_system.get_recent(n)}


# ─────────────────────────────────────────────
# WEBSOCKET (OpenEnv spec)
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent interaction.
    Supports: reset, step, state messages.

    Send JSON: {"action": "reset", "task_id": "easy"}
    Send JSON: {"action": "step", "data": {"action_type": "dispatch", "batch_id": "B001", "truck_id": "T01"}}
    Send JSON: {"action": "state"}
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action_type = msg.get("action", "")

            if action_type == "reset":
                task_id = msg.get("task_id", "easy")
                obs = env.reset(task_id=task_id)
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": 0.0,
                    "done": False,
                })

            elif action_type == "step":
                action_data = msg.get("data", {})
                action = FreshChainAction(**action_data)
                obs = env.step(action)
                await websocket.send_json({
                    "observation": obs.model_dump(),
                    "reward": obs.reward,
                    "done": obs.done,
                    "info": {"task_score": obs.task_score},
                })

            elif action_type == "state":
                await websocket.send_json(env.state.model_dump())

            else:
                await websocket.send_json({"error": f"Unknown action: {action_type}"})

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
