"""
FreshChain Post-Harvest Yield Loss Environment
models.py — Typed Observation, Action, and State models (Pydantic)

OpenEnv Specification:
  - Observation  → what the agent SEES each step
  - Action       → what the agent CAN DO each step
  - State        → internal episode metadata

Real-world context:
  India loses 30-40% of fruits & vegetables post-harvest (~₹92,000 crore/year).
  This environment simulates a cold-chain warehouse where an AI agent
  makes logistics decisions to minimize spoilage and maximize yield saved.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class BatchInfo(BaseModel):
    """A single produce batch currently in the warehouse."""
    batch_id: str = Field(description="Unique identifier e.g. 'B001'")
    crop_type: str = Field(description="Crop name: tomato, potato, onion, banana, etc.")
    quantity_kg: float = Field(description="Remaining quantity in kilograms")
    temperature_c: float = Field(description="Storage temperature in Celsius (ideal: 8-12C)")
    humidity_pct: float = Field(description="Storage humidity percentage (ideal: 55-70%)")
    spoilage_risk: float = Field(
        description="Spoilage risk: 0.0=perfectly fresh, 1.0=fully spoiled. "
                    "Increases every step based on crop type, temperature, and humidity."
    )
    days_in_storage: int = Field(description="Days this batch has been stored")
    market_price_per_kg: float = Field(description="Current APMC market price in INR/kg")


class TruckInfo(BaseModel):
    """A transport vehicle available for dispatching produce to market."""
    truck_id: str = Field(description="Unique identifier e.g. 'T01'")
    capacity_kg: float = Field(description="Maximum cargo capacity in kilograms")
    available: bool = Field(description="True if the truck can accept a dispatch right now")
    destination: str = Field(description="Target APMC market name for this truck")


class FreshChainObservation(BaseModel):
    """
    Full observation returned to the agent after each step (and after reset).
    This is everything the agent can see about the warehouse world.

    Key decision fields:
      - batches       -> which produce needs attention (check spoilage_risk!)
      - trucks        -> which transport is available
      - reward        -> feedback for the last action taken
      - task_score    -> current episode score 0.0-1.0
      - done          -> True = episode over
    """
    step: int = Field(description="Current step number in the episode (starts at 0)")
    batches: List[BatchInfo] = Field(description="All produce batches currently in storage")
    trucks: List[TruckInfo] = Field(description="All trucks (available and unavailable)")
    total_yield_saved_kg: float = Field(
        description="Cumulative kg successfully dispatched to market this episode"
    )
    total_yield_lost_kg: float = Field(
        description="Cumulative kg lost to spoilage or discarded this episode"
    )
    storage_capacity_used_pct: float = Field(
        description="Percentage of warehouse storage currently used (0-100)"
    )
    message: str = Field(description="Human-readable summary of what happened last step")
    done: bool = Field(description="True if the episode has ended")
    reward: float = Field(
        description="Reward for the last action. Positive = good decision, negative = poor."
    )
    task_score: float = Field(
        description="Current task score: saved_kg / (saved_kg + lost_kg), range 0.0-1.0"
    )


class FreshChainAction(BaseModel):
    """
    The action the agent takes each step. One action per step.

    action_type options:
      - 'dispatch'  : Send a batch to market via truck. Needs: batch_id + truck_id
      - 'store'     : Wait one step (all batches age, spoilage increases). Needs: nothing
      - 'reroute'   : Change batch destination. Needs: batch_id + destination
      - 'discard'   : Write off a spoiled batch. Needs: batch_id
    """
    action_type: str = Field(
        description="One of: 'dispatch', 'store', 'reroute', 'discard'"
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Batch to act on. Required for: dispatch, reroute, discard."
    )
    truck_id: Optional[str] = Field(
        default=None,
        description="Truck to use. Required for: dispatch."
    )
    destination: Optional[str] = Field(
        default=None,
        description="New market destination. Required for: reroute."
    )


class FreshChainState(BaseModel):
    """Internal episode tracking metadata. Returned by GET /state."""
    episode_id: str = Field(description="Unique UUID for this episode")
    step_count: int = Field(default=0, description="Steps taken so far")
    task_id: str = Field(description="Task being run: 'easy', 'medium', or 'hard'")
    max_steps: int = Field(description="Maximum steps allowed before episode ends")
