from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models import (
    Action,
    Difficulty,
    Observation,
    RewardBreakdown,
    StateSnapshot,
    StepInfo,
    StepResult,
    TaskSpec,
    TicketView,
)


ALLOWED_CATEGORIES = ["billing", "technical", "account_access", "shipping"]
ALLOWED_PRIORITIES = ["low", "medium", "high"]
ALLOWED_TEMPLATES = [
    "refund_policy",
    "troubleshooting",
    "password_reset",
    "shipping_update",
    "escalation_ack",
    "general_reply",
]


@dataclass(frozen=True)
class TicketTruth:
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    hours_open: int
    category: str
    priority: str
    escalate: bool
    template: str
    sla_target_hours: int
    risk_tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskDefinition:
    spec: TaskSpec
    tickets: List[TicketTruth]


TASKS: Dict[str, TaskDefinition] = {
    "support_triage_easy": TaskDefinition(
        spec=TaskSpec(
            task_id="support_triage_easy",
            title="Retail Support Triage (Easy)",
            difficulty=Difficulty.easy,
            description="Clear keyword-heavy tickets where correct queueing is straightforward.",
            total_tickets=5,
            max_steps=8,
        ),
        tickets=[
            TicketTruth(
                "E-001",
                "Refund for duplicate charge",
                "I was billed twice for order #1928. Please refund one charge.",
                "standard",
                12,
                "billing",
                "medium",
                False,
                "refund_policy",
                24,
                ("revenue",),
            ),
            TicketTruth(
                "E-002",
                "Password reset link expired",
                "I cannot log in. The reset link says expired every time.",
                "business",
                4,
                "account_access",
                "high",
                False,
                "password_reset",
                8,
                ("authentication",),
            ),
            TicketTruth(
                "E-003",
                "Tracking says delivered but missing",
                "Carrier marked delivered but no package at my address.",
                "standard",
                20,
                "shipping",
                "high",
                True,
                "escalation_ack",
                12,
                ("customer_loss",),
            ),
            TicketTruth(
                "E-004",
                "App crashes on checkout",
                "Android app crashes on checkout screen after selecting card.",
                "business",
                6,
                "technical",
                "medium",
                False,
                "troubleshooting",
                12,
                ("checkout",),
            ),
            TicketTruth(
                "E-005",
                "Need invoice copy",
                "Can you send invoice PDF for March billing?",
                "enterprise",
                30,
                "billing",
                "low",
                False,
                "general_reply",
                48,
                ("audit",),
            ),
        ],
    ),
    "support_triage_medium": TaskDefinition(
        spec=TaskSpec(
            task_id="support_triage_medium",
            title="Omnichannel Support Queue (Medium)",
            difficulty=Difficulty.medium,
            description="Ambiguous tickets requiring good priority and template choice under SLA pressure.",
            total_tickets=7,
            max_steps=11,
        ),
        tickets=[
            TicketTruth(
                "M-001",
                "Unexpected annual renewal",
                "We expected monthly billing but got annual charge today.",
                "business",
                10,
                "billing",
                "high",
                False,
                "refund_policy",
                12,
                ("revenue",),
            ),
            TicketTruth(
                "M-002",
                "2FA device lost",
                "I switched phones and cannot receive 2FA codes anymore.",
                "enterprise",
                18,
                "account_access",
                "high",
                True,
                "escalation_ack",
                8,
                ("security", "authentication"),
            ),
            TicketTruth(
                "M-003",
                "Shipment delayed 9 days",
                "Customer event is tomorrow and package still in transit.",
                "standard",
                44,
                "shipping",
                "high",
                True,
                "escalation_ack",
                18,
                ("customer_loss", "time_sensitive"),
            ),
            TicketTruth(
                "M-004",
                "API timeout spikes",
                "POST /checkout endpoint timing out around 20% requests.",
                "enterprise",
                8,
                "technical",
                "high",
                True,
                "escalation_ack",
                6,
                ("outage", "revenue"),
            ),
            TicketTruth(
                "M-005",
                "Coupon not applied",
                "Promo code accepted but invoice total unchanged.",
                "standard",
                5,
                "billing",
                "medium",
                False,
                "general_reply",
                24,
                ("revenue",),
            ),
            TicketTruth(
                "M-006",
                "Cannot update saved card",
                "UI blocks card update with unknown error.",
                "business",
                14,
                "technical",
                "medium",
                False,
                "troubleshooting",
                18,
                ("checkout",),
            ),
            TicketTruth(
                "M-007",
                "Wrong region account lock",
                "Moved country and now account is locked by region policy.",
                "business",
                22,
                "account_access",
                "medium",
                True,
                "escalation_ack",
                12,
                ("compliance", "authentication"),
            ),
        ],
    ),
    "support_triage_hard": TaskDefinition(
        spec=TaskSpec(
            task_id="support_triage_hard",
            title="Enterprise Incident Desk (Hard)",
            difficulty=Difficulty.hard,
            description="Mixed intent, legal risk, and urgent incidents requiring consistent high-quality triage.",
            total_tickets=9,
            max_steps=14,
        ),
        tickets=[
            TicketTruth(
                "H-001",
                "Chargeback threat",
                "Client threatens chargeback for outage credits not reflected yet.",
                "enterprise",
                36,
                "billing",
                "high",
                True,
                "escalation_ack",
                12,
                ("legal", "revenue"),
            ),
            TicketTruth(
                "H-002",
                "Admin account takeover",
                "Unrecognized MFA reset happened and all admins were signed out.",
                "enterprise",
                2,
                "account_access",
                "high",
                True,
                "escalation_ack",
                2,
                ("security", "account_takeover"),
            ),
            TicketTruth(
                "H-003",
                "Bulk order split unexpectedly",
                "Single order was split into three shipments and customs fees tripled.",
                "business",
                28,
                "shipping",
                "medium",
                True,
                "shipping_update",
                24,
                ("revenue", "customer_loss"),
            ),
            TicketTruth(
                "H-004",
                "Intermittent invoice mismatch",
                "Tax lines differ between PDF invoice and dashboard totals.",
                "business",
                40,
                "billing",
                "medium",
                False,
                "general_reply",
                36,
                ("audit", "compliance"),
            ),
            TicketTruth(
                "H-005",
                "Checkout CPU saturation",
                "Pods hit 99% CPU after last deploy and checkout latency doubled.",
                "enterprise",
                3,
                "technical",
                "high",
                True,
                "escalation_ack",
                4,
                ("outage", "checkout"),
            ),
            TicketTruth(
                "H-006",
                "Old password works",
                "Security concern: old password accepted for 30 seconds after change.",
                "enterprise",
                7,
                "account_access",
                "high",
                True,
                "escalation_ack",
                6,
                ("security", "authentication"),
            ),
            TicketTruth(
                "H-007",
                "Where is replacement unit?",
                "RMA approved but replacement tracking never updated.",
                "standard",
                26,
                "shipping",
                "medium",
                False,
                "shipping_update",
                24,
                ("customer_loss",),
            ),
            TicketTruth(
                "H-008",
                "Need partial refund breakdown",
                "Finance requests line-by-line refund explanation for audit.",
                "business",
                16,
                "billing",
                "low",
                False,
                "refund_policy",
                48,
                ("audit",),
            ),
            TicketTruth(
                "H-009",
                "Legacy SDK TLS errors",
                "Payments fail only for clients on legacy SDK v2.3.",
                "business",
                21,
                "technical",
                "medium",
                False,
                "troubleshooting",
                24,
                ("checkout", "revenue"),
            ),
        ],
    ),
}


def _difficulty_weights(difficulty: Difficulty) -> Dict[str, float]:
    if difficulty == Difficulty.easy:
        return {"category": 0.40, "priority": 0.25, "escalation": 0.15, "template": 0.20}
    if difficulty == Difficulty.medium:
        return {"category": 0.35, "priority": 0.25, "escalation": 0.20, "template": 0.20}
    return {"category": 0.30, "priority": 0.20, "escalation": 0.25, "template": 0.25}


def _priority_distance(pred: str, truth: str) -> float:
    levels = {"low": 0, "medium": 1, "high": 2}
    dist = abs(levels[pred] - levels[truth])
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.5
    return 0.0


def _tier_multiplier(customer_tier: str) -> float:
    return {"standard": 1.0, "business": 1.12, "enterprise": 1.28}[customer_tier]


def _risk_band(ticket: TicketTruth) -> str:
    critical_tags = {"security", "outage", "legal", "account_takeover"}
    sensitive_tags = {"audit", "compliance", "revenue", "customer_loss", "authentication"}
    if any(tag in critical_tags for tag in ticket.risk_tags):
        return "critical"
    if ticket.escalate or any(tag in sensitive_tags for tag in ticket.risk_tags):
        return "sensitive"
    return "routine"


def _risk_multiplier(ticket: TicketTruth) -> float:
    band = _risk_band(ticket)
    if band == "critical":
        return 1.35
    if band == "sensitive":
        return 1.15
    return 1.0


def _template_score(ticket: TicketTruth, action: Action) -> float:
    if action.response_template == ticket.template:
        return 1.0
    if action.response_template == "general_reply" and not ticket.escalate:
        return 0.45
    if ticket.category == "billing" and action.response_template in {"refund_policy", "general_reply"}:
        return 0.5
    if ticket.category == "technical" and action.response_template in {"troubleshooting", "escalation_ack"}:
        return 0.55 if ticket.escalate else 0.35
    if ticket.category == "account_access" and action.response_template in {"password_reset", "escalation_ack"}:
        return 0.60
    if ticket.category == "shipping" and action.response_template in {"shipping_update", "escalation_ack"}:
        return 0.60 if ticket.escalate else 0.40
    return 0.0


class SupportTriageEnv:
    def __init__(self) -> None:
        self.task = TASKS["support_triage_easy"]
        self.seed = 7
        self.episode_tickets: List[TicketTruth] = list(self.task.tickets)
        self.step_count = 0
        self.current_index = 0
        self.max_steps = self.task.spec.max_steps
        self.done = False
        self.done_reason = "in_progress"
        self.cumulative_reward = 0.0
        self.processed_tickets: List[Dict[str, object]] = []
        self.last_action_error: Optional[str] = None
        self.deferred_count = 0
        self.consecutive_defers = 0
        self.current_wait_bonus_hours = 0

    def reset(self, task_id: str, seed: int = 7, max_episode_steps: Optional[int] = None) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task = TASKS[task_id]
        self.seed = seed
        self.episode_tickets = list(self.task.tickets)
        if len(self.episode_tickets) > 1:
            random.Random(seed).shuffle(self.episode_tickets)
        self.step_count = 0
        self.current_index = 0
        self.max_steps = max_episode_steps or self.task.spec.max_steps
        self.done = False
        self.done_reason = "in_progress"
        self.cumulative_reward = 0.0
        self.processed_tickets = []
        self.last_action_error = None
        self.deferred_count = 0
        self.consecutive_defers = 0
        self.current_wait_bonus_hours = 0
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if self.done:
            info = self._step_info(self._empty_breakdown())
            info.last_action_error = "Episode already finished. Call reset()."
            self.last_action_error = info.last_action_error
            return StepResult(observation=self._observation(), reward=0.0, done=True, info=info)

        self.step_count += 1
        ticket = self.episode_tickets[self.current_index]
        effective_hours_open = ticket.hours_open + self.current_wait_bonus_hours
        self.last_action_error = None

        if action.defer:
            self.deferred_count += 1
            reward, grader = self._grade_defer(ticket, effective_hours_open)
            self.consecutive_defers += 1
            self.current_wait_bonus_hours += 6
        else:
            self.consecutive_defers = 0
            reward, grader = self._grade_action(ticket, action, effective_hours_open)
            self.processed_tickets.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "observed_hours_open": effective_hours_open,
                    "sla_target_hours": ticket.sla_target_hours,
                    "risk_band": _risk_band(ticket),
                    "predicted": action.model_dump(),
                    "truth": {
                        "category": ticket.category,
                        "priority": ticket.priority,
                        "escalate": ticket.escalate,
                        "template": ticket.template,
                        "risk_tags": list(ticket.risk_tags),
                    },
                    "reward": round(reward, 4),
                    "grader": grader.model_dump(),
                }
            )
            self.current_wait_bonus_hours = 0
            self.current_index += 1

        self.cumulative_reward += reward

        if self.current_index >= len(self.episode_tickets):
            self.done = True
            self.done_reason = "all_tickets_processed"
        elif self.step_count >= self.max_steps:
            self.done = True
            self.done_reason = "max_steps_reached"
            self.last_action_error = "Max steps reached before processing all tickets."

        info = self._step_info(grader)
        info.last_action_error = self.last_action_error
        return StepResult(observation=self._observation(), reward=round(reward, 4), done=self.done, info=info)

    def state(self) -> StateSnapshot:
        normalized_score = self.cumulative_reward / float(self.task.spec.total_tickets)
        normalized_score = max(0.0, min(1.0, normalized_score))
        return StateSnapshot(
            task=self.task.spec,
            seed=self.seed,
            step=self.step_count,
            done=self.done,
            done_reason=self.done_reason,
            cumulative_reward=round(self.cumulative_reward, 4),
            normalized_score=round(normalized_score, 4),
            processed_tickets=self.processed_tickets,
            deferred_count=self.deferred_count,
            consecutive_defers=self.consecutive_defers,
            current_wait_bonus_hours=self.current_wait_bonus_hours,
            last_action_error=self.last_action_error,
        )

    def _empty_breakdown(self) -> RewardBreakdown:
        return RewardBreakdown(
            category_score=0.0,
            priority_score=0.0,
            escalation_score=0.0,
            template_score=0.0,
            accuracy_score=0.0,
            business_impact_penalty=0.0,
            defer_penalty=0.0,
            penalty=0.0,
            step_reward=0.0,
        )

    def _step_info(self, grader: RewardBreakdown) -> StepInfo:
        episode_score = self.cumulative_reward / float(self.task.spec.total_tickets)
        return StepInfo(
            task_id=self.task.spec.task_id,
            difficulty=self.task.spec.difficulty,
            grader=grader,
            episode_score=max(0.0, min(1.0, round(episode_score, 4))),
            done_reason=self.done_reason,
            last_action_error=self.last_action_error,
        )

    def _observation(self) -> Observation:
        current_ticket: Optional[TicketView] = None
        if not self.done and self.current_index < len(self.episode_tickets):
            ticket = self.episode_tickets[self.current_index]
            current_ticket = TicketView(
                ticket_id=ticket.ticket_id,
                subject=ticket.subject,
                body=ticket.body,
                customer_tier=ticket.customer_tier,  # type: ignore[arg-type]
                hours_open=ticket.hours_open + self.current_wait_bonus_hours,
                sla_target_hours=ticket.sla_target_hours,
                risk_band=_risk_band(ticket),  # type: ignore[arg-type]
            )

        completed = min(self.current_index, self.task.spec.total_tickets)
        remaining = max(0, self.task.spec.total_tickets - completed)
        progress = completed / float(self.task.spec.total_tickets)
        return Observation(
            task_id=self.task.spec.task_id,
            step=self.step_count,
            max_steps=self.max_steps,
            total_tickets=self.task.spec.total_tickets,
            completed_tickets=completed,
            remaining_tickets=remaining,
            progress=round(progress, 4),
            current_ticket=current_ticket,
            allowed_categories=ALLOWED_CATEGORIES,
            allowed_priorities=ALLOWED_PRIORITIES,
            allowed_templates=ALLOWED_TEMPLATES,
            last_action_error=self.last_action_error,
        )

    def _sla_pressure(self, ticket: TicketTruth, effective_hours_open: int) -> float:
        return min(2.0, effective_hours_open / float(ticket.sla_target_hours))

    def _business_penalty(self, ticket: TicketTruth, action: Action, effective_hours_open: int) -> float:
        tier_multiplier = _tier_multiplier(ticket.customer_tier)
        risk_multiplier = _risk_multiplier(ticket)
        sla_pressure = self._sla_pressure(ticket, effective_hours_open)
        penalty = 0.0

        if action.escalate and not ticket.escalate:
            penalty += 0.08 * tier_multiplier
        if ticket.escalate and not action.escalate:
            penalty += 0.18 * tier_multiplier * risk_multiplier
        if action.priority == "low" and sla_pressure >= 1.0:
            penalty += 0.10 * tier_multiplier * min(1.5, sla_pressure)
        elif action.priority == "medium" and ticket.priority == "high" and sla_pressure >= 1.0:
            penalty += 0.05 * risk_multiplier
        if any(tag in ticket.risk_tags for tag in ("security", "outage", "legal", "account_takeover")) and not action.escalate:
            penalty += 0.12
        if ticket.customer_tier == "enterprise" and action.response_template == "general_reply":
            penalty += 0.05
        if len(action.note.strip()) > 220:
            penalty += 0.03

        return min(1.0, penalty)

    def _grade_action(self, ticket: TicketTruth, action: Action, effective_hours_open: int) -> tuple[float, RewardBreakdown]:
        weights = _difficulty_weights(self.task.spec.difficulty)
        category_score = 1.0 if action.category == ticket.category else 0.0
        priority_score = _priority_distance(action.priority, ticket.priority)
        escalation_score = 1.0 if action.escalate == ticket.escalate else 0.0
        template_score = _template_score(ticket, action)

        accuracy_score = (
            weights["category"] * category_score
            + weights["priority"] * priority_score
            + weights["escalation"] * escalation_score
            + weights["template"] * template_score
        )
        business_penalty = self._business_penalty(ticket, action, effective_hours_open)
        reward = max(0.0, min(1.0, accuracy_score - business_penalty))

        breakdown = RewardBreakdown(
            category_score=round(category_score, 4),
            priority_score=round(priority_score, 4),
            escalation_score=round(escalation_score, 4),
            template_score=round(template_score, 4),
            accuracy_score=round(accuracy_score, 4),
            business_impact_penalty=round(business_penalty, 4),
            defer_penalty=0.0,
            penalty=round(business_penalty, 4),
            step_reward=round(reward, 4),
        )
        return reward, breakdown

    def _grade_defer(self, ticket: TicketTruth, effective_hours_open: int) -> tuple[float, RewardBreakdown]:
        sla_pressure = self._sla_pressure(ticket, effective_hours_open)
        tier_multiplier = _tier_multiplier(ticket.customer_tier)
        risk_multiplier = _risk_multiplier(ticket)
        repeat_penalty = 0.06 * self.consecutive_defers

        base_credit = 0.04 if ticket.priority == "low" and not ticket.escalate and sla_pressure < 0.6 else 0.0
        defer_penalty = 0.03 * tier_multiplier + 0.05 * sla_pressure + repeat_penalty
        if ticket.escalate:
            defer_penalty += 0.16 * risk_multiplier
        if _risk_band(ticket) == "critical":
            defer_penalty += 0.10

        reward = max(0.0, min(1.0, base_credit - defer_penalty))
        breakdown = RewardBreakdown(
            category_score=0.0,
            priority_score=0.0,
            escalation_score=0.0,
            template_score=0.0,
            accuracy_score=0.0,
            business_impact_penalty=0.0,
            defer_penalty=round(min(1.0, defer_penalty), 4),
            penalty=round(min(1.0, defer_penalty), 4),
            step_reward=round(reward, 4),
        )
        return reward, breakdown
