from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from models import (
    Action,
    MetadataResponse,
    Observation,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    StateResponse,
    StepResult,
    TasksResponse,
)
from server.simulation import SupportTriageEnv, TASKS


APP_VERSION = "1.1.0"
BENCHMARK = "customer_support_triage_openenv"

app = FastAPI(
    title="Customer Support Triage - OpenEnv",
    version=APP_VERSION,
    description=(
        "A real-world simulation where an agent triages customer support tickets. "
        "The agent must predict category, priority, escalation, and response template."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SupportTriageEnv()


UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Customer Support Triage UI</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: #fffaf3;
      --panel-strong: #fff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e8dcc9;
      --accent: #006d77;
      --accent-soft: #d7f0eb;
      --warn: #b42318;
      --ok: #12715b;
      --shadow: 0 14px 35px rgba(31, 41, 55, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(0, 109, 119, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(238, 155, 0, 0.12), transparent 24%),
        linear-gradient(180deg, #fbf7f1 0%, var(--bg) 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 22px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 3vw, 42px);
      line-height: 1.05;
    }

    .hero p {
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 15px;
    }

    .badge {
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(0, 109, 119, 0.18);
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }

    .panel {
      background: rgba(255, 250, 243, 0.9);
      backdrop-filter: blur(6px);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 18px 20px 12px;
    }

    .panel-header h2 {
      margin: 0;
      font-size: 21px;
    }

    .panel-body {
      padding: 0 20px 20px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .stat {
      padding: 14px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }

    .stat-label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 24px;
      font-weight: 700;
    }

    .controls {
      display: grid;
      grid-template-columns: 1.4fr 0.6fr 0.8fr auto;
      gap: 12px;
      margin-bottom: 16px;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
    }

    input, select, textarea, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid #d8ccb7;
      font: inherit;
    }

    input, select, textarea {
      padding: 12px 13px;
      background: white;
      color: var(--ink);
    }

    textarea {
      min-height: 96px;
      resize: vertical;
    }

    button {
      padding: 12px 14px;
      font-weight: 700;
      background: var(--accent);
      color: white;
      cursor: pointer;
      transition: transform 0.14s ease, opacity 0.14s ease;
    }

    button:hover { transform: translateY(-1px); }
    button.secondary {
      background: #415a77;
    }
    button.ghost {
      background: #fff;
      color: var(--ink);
    }

    .button-row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }

    .button-row button { flex: 1; }

    .ticket-card, .json-card, .history-item {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }

    .ticket-card {
      margin-bottom: 16px;
    }

    .ticket-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 12px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #f2f4f7;
      color: #344054;
      font-size: 12px;
      font-weight: 700;
    }

    .msg {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid transparent;
      font-size: 14px;
    }

    .msg.info {
      background: #eef7f7;
      color: #145b63;
      border-color: rgba(0, 109, 119, 0.16);
    }

    .msg.error {
      background: #fef3f2;
      color: var(--warn);
      border-color: rgba(180, 35, 24, 0.16);
    }

    .msg.success {
      background: #ecfdf3;
      color: var(--ok);
      border-color: rgba(18, 113, 91, 0.18);
    }

    .subtle {
      color: var(--muted);
      font-size: 13px;
    }

    .history-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 720px;
      overflow: auto;
      padding-right: 2px;
    }

    .history-item h3 {
      margin: 0 0 10px;
      font-size: 15px;
    }

    .history-item pre, .json-card pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      color: #243b53;
    }

    .empty {
      padding: 26px 18px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.55);
    }

    .footer-note {
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }

    @media (max-width: 1080px) {
      .grid { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 700px) {
      .shell { padding: 18px 14px 28px; }
      .hero { flex-direction: column; }
      .controls, .form-grid, .stats { grid-template-columns: 1fr; }
      .button-row { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>Customer Support Triage</h1>
        <p>Run the OpenEnv locally with a proper dashboard: reset tasks, triage tickets, inspect reward breakdowns, and watch score evolve as you step through the episode.</p>
      </div>
      <div class="badge">Local UI on the same FastAPI app</div>
    </section>

    <div class="grid">
      <section class="panel">
        <div class="panel-header">
          <h2>Agent Console</h2>
          <div class="subtle" id="status-line">Waiting for environment reset.</div>
        </div>
        <div class="panel-body">
          <div id="message" class="msg info">Pick a task and reset the environment to begin.</div>

          <div class="stats">
            <div class="stat">
              <span class="stat-label">Step</span>
              <span class="stat-value" id="stat-step">0</span>
            </div>
            <div class="stat">
              <span class="stat-label">Progress</span>
              <span class="stat-value" id="stat-progress">0%</span>
            </div>
            <div class="stat">
              <span class="stat-label">Score</span>
              <span class="stat-value" id="stat-score">0.000</span>
            </div>
            <div class="stat">
              <span class="stat-label">Done</span>
              <span class="stat-value" id="stat-done">No</span>
            </div>
          </div>

          <div class="controls">
            <div>
              <label for="task">Task</label>
              <select id="task">
                <option value="support_triage_easy">support_triage_easy</option>
                <option value="support_triage_medium">support_triage_medium</option>
                <option value="support_triage_hard">support_triage_hard</option>
              </select>
            </div>
            <div>
              <label for="seed">Seed</label>
              <input id="seed" type="number" min="0" value="7" />
            </div>
            <div>
              <label for="max-steps">Max Steps</label>
              <input id="max-steps" type="number" min="1" max="128" placeholder="default" />
            </div>
            <div style="display:flex;align-items:flex-end;">
              <button id="reset-btn" type="button">Reset</button>
            </div>
          </div>

          <div class="ticket-card">
            <div class="subtle">Current ticket</div>
            <h3 id="ticket-subject" style="margin:8px 0 6px;font-size:24px;">No active ticket</h3>
            <div class="ticket-meta" id="ticket-meta"></div>
            <div id="ticket-body" class="subtle">Reset the environment to load the first ticket.</div>
          </div>

          <form id="action-form">
            <div class="form-grid">
              <div>
                <label for="category">Category</label>
                <select id="category" required>
                  <option value="billing">billing</option>
                  <option value="technical">technical</option>
                  <option value="account_access">account_access</option>
                  <option value="shipping">shipping</option>
                </select>
              </div>
              <div>
                <label for="priority">Priority</label>
                <select id="priority" required>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </div>
              <div>
                <label for="template">Response Template</label>
                <select id="template" required>
                  <option value="refund_policy">refund_policy</option>
                  <option value="troubleshooting">troubleshooting</option>
                  <option value="password_reset">password_reset</option>
                  <option value="shipping_update">shipping_update</option>
                  <option value="escalation_ack">escalation_ack</option>
                  <option value="general_reply">general_reply</option>
                </select>
              </div>
              <div>
                <label for="escalate">Escalate</label>
                <select id="escalate">
                  <option value="false">false</option>
                  <option value="true">true</option>
                </select>
              </div>
              <div>
                <label for="note">Agent Note</label>
                <textarea id="note" maxlength="280" placeholder="Short rationale for your triage decision"></textarea>
              </div>
            </div>

            <div class="button-row">
              <button id="step-btn" type="submit">Submit Action</button>
              <button id="defer-btn" class="secondary" type="button">Defer Ticket</button>
              <button id="state-btn" class="ghost" type="button">Refresh State</button>
            </div>
          </form>

          <div class="footer-note">Tip: use <code>/docs</code> if you also want the raw API explorer.</div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <h2>Observer</h2>
          <div class="subtle" id="done-reason">done_reason: in_progress</div>
        </div>
        <div class="panel-body">
          <div class="json-card" style="margin-bottom:16px;">
            <div class="subtle" style="margin-bottom:8px;">Current observation</div>
            <pre id="observation-json">{}</pre>
          </div>

          <div class="json-card" style="margin-bottom:16px;">
            <div class="subtle" style="margin-bottom:8px;">Current state</div>
            <pre id="state-json">{}</pre>
          </div>

          <div>
            <div class="subtle" style="margin-bottom:8px;">Action history</div>
            <div id="history" class="history-list">
              <div class="empty">No actions yet. Your processed tickets and rewards will appear here.</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const els = {
      message: document.getElementById("message"),
      statusLine: document.getElementById("status-line"),
      doneReason: document.getElementById("done-reason"),
      task: document.getElementById("task"),
      seed: document.getElementById("seed"),
      maxSteps: document.getElementById("max-steps"),
      resetBtn: document.getElementById("reset-btn"),
      actionForm: document.getElementById("action-form"),
      category: document.getElementById("category"),
      priority: document.getElementById("priority"),
      template: document.getElementById("template"),
      escalate: document.getElementById("escalate"),
      note: document.getElementById("note"),
      deferBtn: document.getElementById("defer-btn"),
      stateBtn: document.getElementById("state-btn"),
      stepBtn: document.getElementById("step-btn"),
      ticketSubject: document.getElementById("ticket-subject"),
      ticketMeta: document.getElementById("ticket-meta"),
      ticketBody: document.getElementById("ticket-body"),
      statStep: document.getElementById("stat-step"),
      statProgress: document.getElementById("stat-progress"),
      statScore: document.getElementById("stat-score"),
      statDone: document.getElementById("stat-done"),
      observationJson: document.getElementById("observation-json"),
      stateJson: document.getElementById("state-json"),
      history: document.getElementById("history"),
    };

    let currentObservation = null;
    let currentState = null;

    function pretty(data) {
      return JSON.stringify(data ?? {}, null, 2);
    }

    function setMessage(text, type = "info") {
      els.message.className = `msg ${type}`;
      els.message.textContent = text;
    }

    function setBusy(isBusy) {
      [els.resetBtn, els.stepBtn, els.deferBtn, els.stateBtn].forEach((btn) => {
        btn.disabled = isBusy;
        btn.style.opacity = isBusy ? "0.72" : "1";
      });
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `Request failed with ${response.status}`);
      }
      return response.json();
    }

    function renderTicket(observation) {
      const ticket = observation?.current_ticket;
      if (!ticket) {
        els.ticketSubject.textContent = observation?.last_action_error ? "Episode finished" : "No active ticket";
        els.ticketBody.textContent = observation?.last_action_error || "Reset the environment to load the first ticket.";
        els.ticketMeta.innerHTML = "";
        return;
      }

      els.ticketSubject.textContent = ticket.subject;
      els.ticketBody.textContent = ticket.body;
      els.ticketMeta.innerHTML = "";
      [
        `Ticket ${ticket.ticket_id}`,
        `Tier: ${ticket.customer_tier}`,
        `Hours open: ${ticket.hours_open}`,
      ].forEach((text) => {
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = text;
        els.ticketMeta.appendChild(pill);
      });
    }

    function renderHistory(state) {
      const tickets = state?.processed_tickets ?? [];
      if (!tickets.length) {
        els.history.innerHTML = '<div class="empty">No actions yet. Your processed tickets and rewards will appear here.</div>';
        return;
      }

      els.history.innerHTML = "";
      tickets.slice().reverse().forEach((item, reverseIndex) => {
        const card = document.createElement("div");
        card.className = "history-item";
        const stepNumber = tickets.length - reverseIndex;
        card.innerHTML = `
          <h3>Processed ticket ${item.ticket_id} · step ${stepNumber} · reward ${Number(item.reward).toFixed(3)}</h3>
          <pre>${pretty(item)}</pre>
        `;
        els.history.appendChild(card);
      });
    }

    function render(observation, state) {
      currentObservation = observation;
      currentState = state;

      els.observationJson.textContent = pretty(observation);
      els.stateJson.textContent = pretty(state);
      renderTicket(observation);
      renderHistory(state);

      els.statStep.textContent = String(state?.step ?? observation?.step ?? 0);
      els.statProgress.textContent = `${Math.round((observation?.progress ?? 0) * 100)}%`;
      els.statScore.textContent = Number(state?.normalized_score ?? 0).toFixed(3);
      els.statDone.textContent = state?.done ? "Yes" : "No";
      els.doneReason.textContent = `done_reason: ${state?.done_reason ?? "in_progress"}`;
      els.statusLine.textContent = observation?.task_id
        ? `${observation.task_id} · ${observation.completed_tickets}/${observation.total_tickets} tickets processed`
        : "Waiting for environment reset.";
    }

    async function refreshState() {
      const stateResponse = await fetchJson("/state");
      currentState = stateResponse.state;
      render(currentObservation, currentState);
    }

    async function resetEnvironment() {
      setBusy(true);
      try {
        const payload = {
          task_id: els.task.value,
          seed: Number(els.seed.value || 7),
        };
        if (els.maxSteps.value) {
          payload.max_episode_steps = Number(els.maxSteps.value);
        }

        const resetData = await fetchJson("/reset", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        const stateData = await fetchJson("/state");
        render(resetData.observation, stateData.state);
        setMessage(`Environment reset for ${payload.task_id}. Start triaging the current ticket.`, "success");
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        setBusy(false);
      }
    }

    async function submitAction(overrides = {}) {
      if (!currentObservation) {
        setMessage("Reset the environment before submitting an action.", "error");
        return;
      }

      setBusy(true);
      try {
        const payload = {
          category: els.category.value,
          priority: els.priority.value,
          escalate: els.escalate.value === "true",
          response_template: els.template.value,
          defer: false,
          note: els.note.value,
          ...overrides,
        };

        const stepData = await fetchJson("/step", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        const stateData = await fetchJson("/state");
        render(stepData.observation, stateData.state);

        if (stepData.done) {
          setMessage(`Episode complete. Final score: ${Number(stepData.info.episode_score).toFixed(3)}.`, "success");
        } else if (stepData.info.last_action_error) {
          setMessage(stepData.info.last_action_error, "error");
        } else {
          setMessage(`Step reward: ${Number(stepData.reward).toFixed(3)} · running score: ${Number(stepData.info.episode_score).toFixed(3)}`, "info");
        }
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        setBusy(false);
      }
    }

    els.resetBtn.addEventListener("click", resetEnvironment);
    els.stateBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        await refreshState();
        setMessage("State refreshed from the running environment.", "info");
      } catch (error) {
        setMessage(error.message, "error");
      } finally {
        setBusy(false);
      }
    });

    els.actionForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      await submitAction();
    });

    els.deferBtn.addEventListener("click", async () => {
      await submitAction({
        defer: true,
        note: els.note.value,
      });
    });

    refreshState().catch(() => {});
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata", response_model=MetadataResponse, tags=["system"])
def metadata() -> MetadataResponse:
    return MetadataResponse(
        name="Customer Support Triage - OpenEnv",
        description="Programmatic triage benchmark with deterministic graders and dense rewards.",
        benchmark=BENCHMARK,
        version=APP_VERSION,
        rewards_range=[0.0, 1.0],
        tasks=[task.spec for task in TASKS.values()],
    )


@app.get("/schema", response_model=SchemaResponse, tags=["system"])
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=Action.model_json_schema(),
        observation=Observation.model_json_schema(),
        state=StateResponse.model_json_schema(),
    )


@app.get("/tasks", response_model=TasksResponse, tags=["openenv"])
def tasks() -> TasksResponse:
    return TasksResponse(tasks=[task.spec for task in TASKS.values()])


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(req: ResetRequest) -> ResetResponse:
    try:
        observation = env.reset(task_id=req.task_id, seed=req.seed, max_episode_steps=req.max_episode_steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    info: Dict[str, Any] = {
        "task_id": req.task_id,
        "seed": req.seed,
        "max_steps": observation.max_steps,
        "total_tickets": observation.total_tickets,
    }
    return ResetResponse(observation=observation, info=info)


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(action_payload: Dict[str, Any]) -> StepResult:
    try:
        action = Action.model_validate(action_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return env.step(action)


@app.get("/state", response_model=StateResponse, tags=["openenv"])
def state() -> StateResponse:
    return StateResponse(state=env.state())


@app.post("/mcp", tags=["system"])
def mcp(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP tool bridge is not implemented for this benchmark.",
        },
    }


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
