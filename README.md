---
title: Customer Support Triage OpenEnv
emoji: "📨"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - customer-support
  - triage
  - simulation
---

# Customer Support Triage OpenEnv

This project is an OpenEnv benchmark for one of the most common real-world agent tasks: triaging inbound support tickets under time pressure, business pressure, and escalation risk.

Instead of asking an agent to solve a toy puzzle, this environment asks it to make operational decisions that support teams make every day:

- which queue a ticket belongs in
- how urgent it is
- whether it must be escalated
- what response template is safe to use
- whether deferring it is acceptable or harmful

**"A benchmark for AI support agents that measures operational judgment, not just text generation."**

That works well in a hackathon because it has all three things judges tend to reward:

- a clear real-world problem
- a crisp interactive demo
- an evaluation story that is more serious than "the model felt good"

The strongest angle is that the environment turns support operations into a measurable decision-making task with dense rewards, deterministic episodes, and visible business tradeoffs.

## What makes this environment more credible

This environment now behaves more like a proper benchmark:

- episodes are deterministic but seedable, so `reset(seed=...)` actually changes the ticket order
- tickets carry implicit SLA pressure, customer-tier pressure, and risk sensitivity
- deferring a ticket increases visible backlog pressure by aging the ticket
- the reward function combines accuracy with business-impact penalties instead of only checking labels
- security, outage, legal, and enterprise mistakes are treated as more costly than ordinary misses

## OpenEnv Interface

- `GET /` human-friendly local UI
- `POST /reset` start a task episode
- `POST /step` submit one typed action
- `GET /state` inspect full simulator state and normalized score
- `GET /metadata` benchmark metadata
- `GET /schema` JSON schema for action, observation, and state
- `GET /tasks` task catalog
- `GET /health` liveness check
- `GET /docs` FastAPI / OpenAPI explorer

All exposed task and reward scores are normalized to `(0.0, 1.0)` using a small epsilon clamp, so submissions never emit an exact `0.0` or `1.0`.

## Action Space

The agent submits a typed JSON action:

```json
{
  "category": "billing | technical | account_access | shipping",
  "priority": "low | medium | high",
  "escalate": true,
  "response_template": "refund_policy | troubleshooting | password_reset | shipping_update | escalation_ack | general_reply",
  "defer": false,
  "note": "optional short rationale"
}
```

## Observation Space

Each observation includes:

- current ticket text
- customer tier
- current `hours_open`
- `sla_target_hours`
- ticket `risk_band`
- progress through the episode
- allowed action enums
- latest terminal error, if any

## Reward Function

The reward is intentionally not just "exact match or fail."

Each non-defer action gets:

- category correctness
- priority correctness with partial credit for near misses
- escalation correctness
- template correctness with some partial credit for safe-but-generic responses

Then the environment subtracts business-impact penalties for things like:

- missing escalation on security, outage, legal, or enterprise-sensitive tickets
- under-prioritizing tickets that are already at or beyond SLA
- escalating unnecessarily
- using weak generic responses for high-value enterprise cases
- repeatedly deferring live tickets

This means two wrong answers are not equally wrong. Missing escalation on an enterprise account takeover is scored as more harmful than picking a slightly suboptimal template on a routine billing request.

## Tasks

There are three task groups:

1. `support_triage_easy`
2. `support_triage_medium`
3. `support_triage_hard`

They increase in ambiguity, business pressure, and risk sensitivity.

## Local Setup

From the project root:

```bash
pip install -r requirements.txt
python -m server.app
```

Then open:

- `http://localhost:7860/` for the local UI
- `http://localhost:7860/docs` for the API docs

## Hosting 

The fastest public deploy for this repo is **Hugging Face Spaces** because the project already includes [openenv.yaml](hakc/openenv.yaml) and a [Dockerfile](hakc/Dockerfile).

### Hugging Face Spaces

1. Create a new **Docker Space** on Hugging Face.
2. Push this repo to the Space.
3. No required environment variables are needed for the benchmark UI/API itself.
4. Wait for the image build to finish, then open the Space URL.
5. Before submitting to the OpenEnv RL challenge, confirm the Space status is exactly **Running**.
6. If you have other Spaces active, stop the ones you do not need so this primary Space can build and stay live.

The app serves:

- `/` for the interactive demo
- `/docs` for the API explorer
- `/health` for uptime checks



## Baseline Inference

The baseline script is [inference.py](hakc/inference.py).

Environment variables:

- `API_BASE_URL` default: `https://router.huggingface.co/v1`
- `MODEL_NAME` default: `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` required
- `ENV_URL` default: `http://localhost:7860`
- `LOCAL_IMAGE_NAME` declared for compatibility

Run:

```bash
python inference.py
```

The script uses the OpenAI Python client for all LLM calls and expects `HF_TOKEN` to be present. The environment server itself can still run locally without that token.

The stdout contract for each episode is:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

## Validation

```bash
python -m pytest -q
```

## Demo Pitch

1. "Most AI agent demos generate text. Ours makes operational decisions with measurable business consequences."
2. "We built a support-ops benchmark with deterministic tasks, seeded reproducibility, and business-aware rewards."
3. "The same setup can be extended to security triage, trust-and-safety review, claims handling, and IT incident routing."

