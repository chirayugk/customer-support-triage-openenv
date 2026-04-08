from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class TaskSpec(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    description: str
    total_tickets: int
    max_steps: int


class TicketView(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["standard", "business", "enterprise"]
    hours_open: int = Field(ge=0)
    sla_target_hours: int = Field(ge=1)
    risk_band: Literal["routine", "sensitive", "critical"]


class Observation(BaseModel):
    task_id: str
    step: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    total_tickets: int = Field(ge=1)
    completed_tickets: int = Field(ge=0)
    remaining_tickets: int = Field(ge=0)
    progress: float = Field(ge=0.0, le=1.0)
    current_ticket: Optional[TicketView] = None
    allowed_categories: List[str]
    allowed_priorities: List[str]
    allowed_templates: List[str]
    last_action_error: Optional[str] = None


class Action(BaseModel):
    category: Literal["billing", "technical", "account_access", "shipping"]
    priority: Literal["low", "medium", "high"]
    escalate: bool = False
    response_template: Literal[
        "refund_policy",
        "troubleshooting",
        "password_reset",
        "shipping_update",
        "escalation_ack",
        "general_reply",
    ]
    defer: bool = False
    note: str = Field(default="", max_length=280)


class RewardBreakdown(BaseModel):
    category_score: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=1.0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    template_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    business_impact_penalty: float = Field(ge=0.0, le=1.0)
    defer_penalty: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0, le=1.0)
    step_reward: float = Field(ge=0.0, le=1.0)


class StepInfo(BaseModel):
    task_id: str
    difficulty: Difficulty
    grader: RewardBreakdown
    episode_score: float = Field(ge=0.0, le=1.0)
    done_reason: Literal["in_progress", "all_tickets_processed", "max_steps_reached"]
    last_action_error: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: StepInfo


class ResetRequest(BaseModel):
    task_id: str = Field(default="support_triage_easy")
    seed: int = Field(default=7, ge=0)
    max_episode_steps: Optional[int] = Field(default=None, ge=1, le=128)


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, float | str | int]


class StateSnapshot(BaseModel):
    task: TaskSpec
    seed: int = Field(ge=0)
    step: int
    done: bool
    done_reason: str
    cumulative_reward: float = Field(ge=0.0)
    normalized_score: float = Field(ge=0.0, le=1.0)
    processed_tickets: List[Dict[str, object]]
    deferred_count: int = Field(ge=0)
    consecutive_defers: int = Field(ge=0)
    current_wait_bonus_hours: int = Field(ge=0)
    last_action_error: Optional[str] = None


class StateResponse(BaseModel):
    state: StateSnapshot


class MetadataResponse(BaseModel):
    name: str
    description: str
    benchmark: str
    version: str
    rewards_range: List[float]
    tasks: List[TaskSpec]


class SchemaResponse(BaseModel):
    action: Dict[str, object]
    observation: Dict[str, object]
    state: Dict[str, object]


class TasksResponse(BaseModel):
    tasks: List[TaskSpec]
