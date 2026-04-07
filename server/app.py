"""
FastAPI Server for the AI Customer Support OpenEnv Environment.

Exposes:
  POST /reset           → Initialize new episode
  POST /step            → Execute agent action
  GET  /state           → Get current state
  GET  /tasks           → List all tasks
  POST /grade           → Grade agent output (full text)
  GET  /health          → Health check
  GET  /metrics         → Aggregated episode metrics
  GET  /               → Serve premium UI dashboard
"""

from __future__ import annotations
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ─── resolve project root so imports work regardless of cwd ───────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.env import SupportEnv
from app.models import (
    Action, Observation, StepResult, EnvState,
    ResetRequest, GradeRequest, GradeResult, HealthResponse
)
from app.tasks import list_tasks
from app.grader import grade_output
from server.db import create_user, verify_user
from pydantic import BaseModel

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Application Bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support — OpenEnv",
    description=(
        "A reinforcement learning environment for training AI agents to handle "
        "customer support tickets. Implements the standard OpenEnv API: "
        "reset(), step(), state()."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Singleton environment instance (one per running server process) ──────────
env = SupportEnv()

# ─── Server startup time for uptime tracking ──────────────────────────────────
_started_at = time.time()

# ─── Simple metrics store ─────────────────────────────────────────────────────
_metrics: Dict[str, Any] = {
    "total_episodes": 0,
    "total_steps": 0,
    "task_counts": {"task_easy": 0, "task_medium": 0, "task_hard": 0},
    "avg_reward": 0.0,
    "rewards": [],
}


# ---------------------------------------------------------------------------
# Static Files (UI Dashboard)
# ---------------------------------------------------------------------------

UI_DIR = ROOT / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Serve the premium UI dashboard."""
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AI Customer Support OpenEnv</h1><p>UI not found. Visit /docs for API.</p>")

@app.get("/login", response_class=HTMLResponse, tags=["UI"])
async def serve_login():
    """Serve the login UI."""
    login_path = UI_DIR / "login.html"
    if login_path.exists():
        return HTMLResponse(content=login_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Login Page Not Found</h1>")

@app.get("/signup", response_class=HTMLResponse, tags=["UI"])
async def serve_signup():
    """Serve the signup UI."""
    signup_path = UI_DIR / "signup.html"
    if signup_path.exists():
        return HTMLResponse(content=signup_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Sign Up Page Not Found</h1>")

@app.post("/api/signup", tags=["Auth"])
async def api_signup(req: SignupRequest):
    """Register a new user."""
    success = create_user(req.username, req.email, req.password)
    if not success:
        raise HTTPException(status_code=400, detail="Username or email already exists.")
    return {"status": "success", "message": "User created successfully"}

@app.post("/api/login", tags=["Auth"])
async def api_login(req: LoginRequest):
    """Authenticate an existing user."""
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"status": "success", "user": {"id": user["id"], "username": user["username"]}}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint — confirms server is running."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _started_at, 1),
        version=SupportEnv.VERSION,
        environment=SupportEnv.ENV_NAME,
    )


@app.get("/tasks", tags=["Environment"])
async def get_tasks():
    """List all available tasks with their definitions, rewards, and allowed actions."""
    tasks = list_tasks()
    return {"tasks": [t.model_dump() for t in tasks]}


@app.post("/reset", response_model=Observation, tags=["Environment"])
async def reset(body: ResetRequest = None):
    """
    Initialize a new episode.

    - `task_id`: Which task to run (task_easy | task_medium | task_hard)
    - `ticket_index`: Force a specific ticket from the corpus (optional)
    """
    if body is None:
        body = ResetRequest()

    try:
        obs = env.reset(task_id=body.task_id or "task_easy", ticket_index=body.ticket_index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _metrics["total_episodes"] += 1
    _metrics["task_counts"][body.task_id or "task_easy"] = (
        _metrics["task_counts"].get(body.task_id or "task_easy", 0) + 1
    )
    return obs


@app.post("/step", response_model=StepResult, tags=["Environment"])
async def step(action: Action):
    """
    Execute an agent action in the current episode.

    Action types:
    - `classify`  — Classify the ticket into a category
    - `respond`   — Generate a response to the ticket
    - `escalate`  — Escalate to senior support
    - `close`     — Mark ticket as closed

    Returns the new observation, reward, done flag, and info.
    """
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _metrics["total_steps"] += 1
    _metrics["rewards"].append(result.reward)
    if len(_metrics["rewards"]) > 0:
        _metrics["avg_reward"] = round(
            sum(_metrics["rewards"][-100:]) / len(_metrics["rewards"][-100:]), 3
        )
    return result


@app.get("/state", response_model=EnvState, tags=["Environment"])
async def state():
    """Return the full current episode state snapshot."""
    return env.state()


@app.post("/grade", response_model=GradeResult, tags=["Grader"])
async def grade(request: GradeRequest):
    """
    Grade a full agent output string against the task criteria.
    Deterministic scoring between 0.0 and 1.0.

    - `output`:  Combined text of all agent actions
    - `task_id`: Which task to grade against
    """
    current_state = env.state()
    ticket_index = getattr(env, "_ticket_index", 0)
    result = grade_output(request.output, request.task_id, ticket_index)
    return result


@app.get("/metrics", tags=["System"])
async def metrics():
    """Return aggregated runtime metrics for the environment."""
    return {
        **_metrics,
        "uptime_seconds": round(time.time() - _started_at, 1),
    }


@app.get("/tickets", tags=["Environment"])
async def list_tickets():
    """List all ticket examples in the corpus (for development/testing)."""
    from app.tasks import TICKET_CORPUS
    return {
        "count": len(TICKET_CORPUS),
        "tickets": [
            {"index": i, "text": t["text"], "category": t["category"]}
            for i, t in enumerate(TICKET_CORPUS)
        ],
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)
