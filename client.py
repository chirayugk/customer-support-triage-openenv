from __future__ import annotations

from typing import Optional

import httpx

from models import (
    Action,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepResult,
    TasksResponse,
)


class SupportOpenEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def tasks(self) -> TasksResponse:
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return TasksResponse.model_validate(resp.json())

    def reset(
        self,
        task_id: str = "support_triage_easy",
        seed: int = 7,
        max_episode_steps: Optional[int] = None,
    ) -> ResetResponse:
        req = ResetRequest(task_id=task_id, seed=seed, max_episode_steps=max_episode_steps)
        resp = self._client.post("/reset", json=req.model_dump(exclude_none=True))
        resp.raise_for_status()
        return ResetResponse.model_validate(resp.json())

    def step(self, action: Action) -> StepResult:
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> StateResponse:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return StateResponse.model_validate(resp.json())

    def close(self) -> None:
        self._client.close()
