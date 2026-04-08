from fastapi.testclient import TestClient

from models import Action
from server.app import app
from server.simulation import SupportTriageEnv, TASKS


def test_three_tasks_available():
    assert len(TASKS) >= 3
    assert {"support_triage_easy", "support_triage_medium", "support_triage_hard"}.issubset(TASKS.keys())


def test_reset_and_state_shape():
    env = SupportTriageEnv()
    obs = env.reset("support_triage_easy")
    assert obs.total_tickets == 5
    state = env.state()
    assert state.normalized_score == 0.0
    assert state.done is False
    assert state.seed == 7


def test_seeded_order_is_reproducible():
    env1 = SupportTriageEnv()
    env2 = SupportTriageEnv()
    env1.reset("support_triage_medium", seed=11)
    env2.reset("support_triage_medium", seed=11)
    order1 = [ticket.ticket_id for ticket in env1.episode_tickets]
    order2 = [ticket.ticket_id for ticket in env2.episode_tickets]
    assert order1 == order2


def test_different_seed_changes_ticket_order():
    env1 = SupportTriageEnv()
    env2 = SupportTriageEnv()
    env1.reset("support_triage_hard", seed=1)
    env2.reset("support_triage_hard", seed=2)
    order1 = [ticket.ticket_id for ticket in env1.episode_tickets]
    order2 = [ticket.ticket_id for ticket in env2.episode_tickets]
    assert order1 != order2


def test_step_reward_in_range():
    env = SupportTriageEnv()
    env.reset("support_triage_medium")
    action = Action(
        category="billing",
        priority="medium",
        escalate=False,
        response_template="general_reply",
        defer=False,
        note="test",
    )
    result = env.step(action)
    assert 0.0 <= result.reward <= 1.0
    assert 0.0 <= result.info.episode_score <= 1.0


def test_episode_completes():
    env = SupportTriageEnv()
    env.reset("support_triage_easy")
    while not env.done:
        result = env.step(
            Action(
                category="technical",
                priority="low",
                escalate=False,
                response_template="general_reply",
                defer=False,
                note="",
            )
        )
        if result.done:
            break
    assert env.state().done is True


def test_defer_penalty_reduces_reward():
    env = SupportTriageEnv()
    env.reset("support_triage_hard")
    r1 = env.step(
        Action(
            category="technical",
            priority="medium",
            escalate=False,
            response_template="general_reply",
            defer=True,
            note="need more context",
        )
    ).reward
    r2 = env.step(
        Action(
            category="technical",
            priority="medium",
            escalate=False,
            response_template="general_reply",
            defer=True,
            note="still waiting",
        )
    ).reward
    assert r2 <= r1


def test_defer_increases_visible_wait_time():
    env = SupportTriageEnv()
    obs = env.reset("support_triage_easy")
    initial_hours = obs.current_ticket.hours_open
    result = env.step(
        Action(
            category="technical",
            priority="medium",
            escalate=False,
            response_template="general_reply",
            defer=True,
            note="need more context",
        )
    )
    assert result.observation.current_ticket.hours_open == initial_hours + 6


def test_reset_endpoint_accepts_missing_body():
    client = TestClient(app)
    response = client.post("/reset")
    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["task_id"] == "support_triage_easy"
    assert payload["info"]["seed"] == 7
