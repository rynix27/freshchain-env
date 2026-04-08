---
title: FreshChain Post-Harvest Yield Loss Environment
colorFrom: green
colorTo: yellow
sdk: docker
pinned: true
tags:
  - openenv
  - agriculture
  - supply-chain
  - post-harvest
  - india
  - cold-chain
  - rl-environment
---

#  FreshChain — Post-Harvest Yield Loss Prevention Environment

> **Meta PyTorch OpenEnv Hackathon × SST India AI Hackathon 2026**  
> An OpenEnv-compatible Reinforcement Learning environment for real-world agricultural logistics.

---

##  Overview & Motivation

India loses **30–40% of all fruits and vegetables post-harvest** — roughly **₹92,000 crore per year**. Around 60% of this loss is caused by poor logistics decisions: wrong timing, wrong routing, missed markets.

**FreshChain** turns this into an RL problem.

An AI agent controls a cold-chain warehouse and must make real-time decisions about produce batches: when to dispatch them to market, when to wait, when to reroute to a better mandi, and when to discard to prevent a spoilage cascade. The agent learns from dense reward signals at every step — not just a final score.

**This is not a game or toy problem.** Every element — crop spoilage rates, APMC market prices, truck capacities, warehouse conditions — is modeled on real Indian agricultural data.

---

##  What Is an RL Environment?

This is a **Reinforcement Learning environment** following the OpenEnv specification:

```
┌──────────────────────────────────────────────────────────────────┐
│                    FreshChain RL Loop                            │
│                                                                  │
│   ┌────────┐    Observation     ┌─────────┐                      │
│   │  ENV   │ ─────────────────> │  AGENT  │                      │
│   │        │                    │ (LLM or │                      │
│   │ • crops│ <───────────────── │  model) │                      │
│   │ • temp │      Action        └─────────┘                      │
│   │ • price│                                                      │
│   │ • risk │ ─── Reward ──────> Agent updates                    │
│   └────────┘                                                      │
│                                                                  │
│  Each step:  observe → decide → act → receive reward → repeat   │
└──────────────────────────────────────────────────────────────────┘
```

The agent interacts with the environment step-by-step until the episode ends, accumulating rewards based on its decisions.

---

##  Observation Space — What the Agent Sees

After every action, the agent receives a **FreshChainObservation** (Pydantic model):

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current step number |
| `batches` | list[BatchInfo] | All produce batches in storage |
| `trucks` | list[TruckInfo] | All transport vehicles |
| `total_yield_saved_kg` | float | Cumulative kg dispatched to market |
| `total_yield_lost_kg` | float | Cumulative kg lost to spoilage |
| `storage_capacity_used_pct` | float | % of warehouse used |
| `message` | str | What happened last step |
| `done` | bool | True = episode over |
| `reward` | float | Reward for last action |
| `task_score` | float | Running score 0.0–1.0 |

### BatchInfo fields (per batch):

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | str | e.g. "B001" |
| `crop_type` | str | tomato, potato, onion, banana, mango, etc. |
| `quantity_kg` | float | Remaining quantity in kg |
| `temperature_c` | float | Storage temp (above 15°C = faster spoilage) |
| `humidity_pct` | float | Humidity (above 70% = faster spoilage) |
| **`spoilage_risk`** | float | **KEY: 0.0=fresh → 1.0=fully spoiled** |
| `days_in_storage` | int | Days warehoused |
| `market_price_per_kg` | float | Current APMC price in INR/kg |

---

##  Action Space — What the Agent Can Do

The agent picks **one action per step**:

| `action_type` | Required fields | Effect |
|---------------|----------------|--------|
| `dispatch` | `batch_id` + `truck_id` | Sends batch to market → yield saved |
| `store` | nothing | Wait; all batches age → spoilage_risk increases |
| `reroute` | `batch_id` + `destination` | Change market destination → may get better price |
| `discard` | `batch_id` | Write off batch; prevents cascade spoilage |

**Example action JSON:**
```json
{
  "action_type": "dispatch",
  "batch_id": "B001",
  "truck_id": "T01"
}
```

---

##  Reward Function — Feedback at Every Step

The reward is **dense**: the agent receives feedback after every action, not just at the end.

| Situation | Reward |
|-----------|--------|
| Dispatch a batch successfully | `+0.1 to +2.0` (proportional to quantity saved) |
| Dispatching a HIGH-risk batch (risk > 0.6) | `+0.2 to +0.3` bonus |
| Rerouting to better market | `+0.1` |
| Discarding critical batch (risk > 0.75) | `+0.05` (prevents cascade) |
| Storing (waiting one step) | `-0.05` (small time penalty) |
| Spoilage event this step | `-0.2` |
| Invalid action (wrong batch/truck ID) | `-0.15` |
| Discarding a healthy batch | `-0.3` (waste penalty) |

**Final task score formula:**
```
base_score       = yield_saved_kg / (yield_saved_kg + yield_lost_kg)
efficiency_bonus = max(0, (max_steps - steps_used) / max_steps) × 0.1
final_score      = min(1.0, base_score + efficiency_bonus)
```

---

##  Tasks — Three Levels of Difficulty

### Task 1: Easy — Basic Spoilage Control
| Property | Value |
|----------|-------|
| Batches | 1 |
| Trucks | 1 |
| Special events | None |
| Max steps | 5 |
| **Objective** | Dispatch the single batch before spoilage_risk reaches 1.0 |
| **Pass threshold** | Score ≥ 0.80 |
| **Expected score** | 0.80–1.00 |

**What this tests:** Can the agent learn the basic dispatch-or-wait decision?

---

### Task 2: Medium — Prioritization Under Breakdown
| Property | Value |
|----------|-------|
| Batches | 3 |
| Trucks | 2 |
| Special events | Truck T01 breaks down at step 3 |
| Max steps | 10 |
| **Objective** | Dispatch highest-risk batches before the truck breakdown |
| **Pass threshold** | Score ≥ 0.60 |
| **Expected score** | 0.55–0.75 |

**What this tests:** Can the agent prioritize urgency under resource constraints?

---

### Task 3: Hard — Dynamic Multi-Objective Optimization
| Property | Value |
|----------|-------|
| Batches | 6 |
| Trucks | 2 |
| Special events | Breakdown at step 3 + price volatility every step |
| Max steps | 15 |
| **Objective** | Maximize yield while handling cascading spoilage, breakdowns, and price changes |
| **Pass threshold** | Score ≥ 0.40 |
| **Expected score** | 0.35–0.55 |

**What this tests:** Multi-objective planning under real-world dynamic uncertainty.

---

##  Graders — Programmatic Scoring (0.0 to 1.0)

Each task has a deterministic grader at `POST /grade`:

```python
def grade(self) -> float:
    total = yield_saved + yield_lost
    if total == 0:
        return 0.0
    base_score = yield_saved / total
    efficiency_bonus = max(0, (max_steps - steps_used) / max_steps) * 0.1
    return round(min(1.0, base_score + efficiency_bonus), 3)
```

**Grading is:**
-  **Deterministic** — same episode state always gives the same score
-  **Reproducible** — no randomness in the scoring logic
-  **Incremental** — task_score updates every step (not just at end)
-  **Partial credit** — saving 70% of yield scores 0.70, not 0

---

##  Real-World Data

All data is based on actual Indian agricultural markets:

**APMC Market Prices (2024–25):**
| Crop | Price Range | Spoilage Speed |
|------|------------|----------------|
| Spinach | ₹10–28/kg | Very fast (0.22/step) |
| Tomato | ₹12–45/kg | Fast (0.18/step) |
| Banana | ₹15–40/kg | Fast (0.15/step) |
| Mango | ₹30–120/kg | Medium (0.14/step) |
| Cauliflower | ₹8–25/kg | Medium (0.12/step) |
| Carrot | ₹12–30/kg | Slow (0.08/step) |
| Potato | ₹8–22/kg | Slow (0.06/step) |
| Onion | ₹10–35/kg | Very slow (0.05/step) |

**Target APMC Markets:**
- Azadpur Mandi (Delhi)
- Vashi APMC (Mumbai)
- Koyambedu (Chennai)
- Gultekdi Market (Pune)
- Bowenpally (Hyderabad)
- Yeshwanthpur (Bangalore)

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Environment info and links |
| GET | `/health` | Health check |
| GET | `/web` | Live warehouse dashboard UI |
| POST | `/reset?task_id=easy` | Start new episode (easy/medium/hard) |
| POST | `/step` | Agent takes one action |
| GET | `/state` | Episode metadata |
| POST | `/grade` | Final score 0.0–1.0 |
| GET | `/tasks` | Task descriptions and grader info |
| GET | `/alerts` | WhatsApp alert log |
| WS | `/ws` | WebSocket real-time interface |

---

##  Setup & Usage

### Docker (local)

```bash
# Build
docker build -t freshchain-env .

# Run
docker run -p 7860:7860 freshchain-env

# Open dashboard
open http://localhost:7860/web
```

### Run baseline agent

```bash
export HF_TOKEN="your_huggingface_or_openai_token"
export MODEL_NAME="gpt-4o-mini"
export API_BASE_URL="https://api.openai.com/v1"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### Manual API test

```bash
# Start episode
curl -X POST "http://localhost:7860/reset?task_id=easy"

# Take an action
curl -X POST "http://localhost:7860/step" \
     -H "Content-Type: application/json" \
     -d '{"action_type": "dispatch", "batch_id": "B001", "truck_id": "T01"}'

# Get final score
curl -X POST "http://localhost:7860/grade"
```

---

##  Baseline Performance Scores

Measured with `gpt-4o-mini` as the agent:

| Task | Score | Threshold | Status |
|------|-------|-----------|--------|
| easy | ~0.87 | 0.80 |  PASS |
| medium | ~0.63 | 0.60 |  PASS |
| hard | ~0.42 | 0.40 | PASS |
| **Average** | **~0.64** | — | — |

---

##  Project Structure

```
freshchain-env/
├── models.py                  # Pydantic models: Action, Observation, State
├── inference.py               # Baseline LLM agent script
├── openenv.yaml               # OpenEnv specification metadata
├── Dockerfile                 # HF Spaces container
├── README.md                  # This file
├── __init__.py
└── server/
    ├── environment.py         # RL simulation (spoilage, trucks, APMC data)
    ├── app.py                 # FastAPI server (OpenEnv HTTP interface)
    ├── whatsapp_alerts.py     # WhatsApp alert simulation
    ├── requirements.txt       # Python dependencies
    ├── __init__.py
    └── static/
        └── index.html         # Live warehouse dashboard UI
```

---

##  Deploy to Hugging Face Spaces

1. Create a new Space at huggingface.co/spaces
2. Set SDK to `Docker`
3. Push this repository
4. Space auto-builds and starts on port 7860

The app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/freshchain-env`

---

*Built for Meta PyTorch OpenEnv Hackathon 2026 · Tags: `openenv` `agriculture` `supply-chain` `post-harvest` `india` `apmc` `cold-chain` `food-security`*
