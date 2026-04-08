import json
import os
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "customer_support_triage_openenv"
TASKS = ["support_triage_easy", "support_triage_medium", "support_triage_hard"]
MAX_OUTPUT_TOKENS = 200

VALID_CATEGORIES = {"billing", "technical", "account_access", "shipping"}
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_TEMPLATES = {
    "refund_policy",
    "troubleshooting",
    "password_reset",
    "shipping_update",
    "escalation_ack",
    "general_reply",
}


def single_line(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ")


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={single_line(task)} env={single_line(env)} model={single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = single_line(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={single_line(action)} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_action(ticket: Dict[str, object]) -> Dict[str, object]:
    text = f"{ticket.get('subject','')} {ticket.get('body','')}".lower()
    if any(k in text for k in ["password", "login", "2fa", "account", "signed out"]):
        category = "account_access"
        template = "password_reset"
    elif any(k in text for k in ["invoice", "refund", "billed", "charge", "billing", "chargeback"]):
        category = "billing"
        template = "refund_policy"
    elif any(k in text for k in ["ship", "tracking", "delivery", "package", "rma", "replacement"]):
        category = "shipping"
        template = "shipping_update"
    else:
        category = "technical"
        template = "troubleshooting"

    hours_open = int(ticket.get("hours_open", 0))
    priority = "high" if hours_open >= 24 else ("medium" if hours_open >= 8 else "low")
    escalate = priority == "high" or any(k in text for k in ["takeover", "outage", "threat", "security"])
    if escalate:
        template = "escalation_ack"

    return {
        "category": category,
        "priority": priority,
        "escalate": escalate,
        "response_template": template,
        "defer": False,
        "note": "heuristic fallback",
    }


def sanitize_action(action: Dict[str, object], ticket: Dict[str, object]) -> Dict[str, object]:
    out = dict(action)
    if str(out.get("category")) not in VALID_CATEGORIES:
        out["category"] = heuristic_action(ticket)["category"]
    if str(out.get("priority")) not in VALID_PRIORITIES:
        out["priority"] = heuristic_action(ticket)["priority"]
    if str(out.get("response_template")) not in VALID_TEMPLATES:
        out["response_template"] = heuristic_action(ticket)["response_template"]
    out["escalate"] = bool(out.get("escalate", False))
    out["defer"] = bool(out.get("defer", False))
    out["note"] = str(out.get("note", ""))[:280]
    return out


def call_model(client: OpenAI, observation: Dict[str, object]) -> Dict[str, object]:
    ticket = observation.get("current_ticket") or {}
    sys_prompt = (
        "You are a customer-support triage agent. "
        "Return exactly one compact JSON object with keys: "
        "category, priority, escalate, response_template, defer, note. "
        "Valid category: billing|technical|account_access|shipping. "
        "Valid priority: low|medium|high. "
        "Valid response_template: refund_policy|troubleshooting|password_reset|shipping_update|escalation_ack|general_reply. "
        "Set defer=false unless absolutely necessary."
    )
    user_prompt = json.dumps(
        {
            "task_id": observation.get("task_id"),
            "step": observation.get("step"),
            "progress": observation.get("progress"),
            "ticket": ticket,
        },
        ensure_ascii=True,
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        action = {
            "category": data.get("category", "technical"),
            "priority": data.get("priority", "medium"),
            "escalate": bool(data.get("escalate", False)),
            "response_template": data.get("response_template", "general_reply"),
            "defer": bool(data.get("defer", False)),
            "note": str(data.get("note", ""))[:280],
        }
        return sanitize_action(action, ticket if isinstance(ticket, dict) else {})
    except Exception:
        return heuristic_action(ticket if isinstance(ticket, dict) else {})


def run_task(client: OpenAI, task_id: str) -> None:
    http = httpx.Client(base_url=ENV_URL.rstrip("/"), timeout=5)
    rewards: List[float] = []
    steps = 0
    score = 0.0
    success = False
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        reset_resp = http.post("/reset", json={"task_id": task_id, "seed": 7})
        reset_resp.raise_for_status()
        obs = reset_resp.json()["observation"]
        done = False
        while not done:
            action = call_model(client, obs)
            step_resp = http.post("/step", json=action)
            if step_resp.status_code == 422:
                action = heuristic_action(obs.get("current_ticket") or {})
                step_resp = http.post("/step", json=action)
            step_resp.raise_for_status()
            payload = step_resp.json()
            obs = payload["observation"]
            reward = float(payload["reward"])
            done = bool(payload["done"])
            info = payload.get("info", {})
            error = info.get("last_action_error")
            steps += 1
            rewards.append(reward)
            action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=True)
            log_step(step=steps, action=action_str, reward=reward, done=done, error=error)
            if steps >= int(obs.get("max_steps", 32)):
                break

        state_resp = http.get("/state")
        state_resp.raise_for_status()
        score = float(state_resp.json()["state"]["normalized_score"])
        score = max(0.0001, min(0.9999, score))
        success = score >= 0.70
    except Exception:
        success = False
        score = 0.0001
    finally:
        http.close()
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=1.5, max_retries=0)
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
