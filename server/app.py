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
from typing import Dict, Any, Optional

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
from openai import OpenAI

try:
    llm_client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("API_KEY", "dummy_key")
    )
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
except Exception as e:
    llm_client = None

class GenerateTicketRequest(BaseModel):
    category: str
    domain: Optional[str] = "general"
    difficulty: Optional[str] = "normal"
    sentiment: Optional[str] = "neutral"
    noise_level: Optional[float] = 0.0

class AnalyzeRequest(BaseModel):
    text: str

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


@app.post("/api/generate_ticket", tags=["GenAI"])
async def generate_ticket(req: GenerateTicketRequest):
    """Dynamically generate a customer support ticket using Gen AI."""
    if not llm_client:
        raise HTTPException(status_code=503, detail="Gen AI client not configured.")
    try:
        sys_prompt = (
            f"You are a customer support ticket generator for the {req.domain} domain. "
            f"Generate a realistic ({req.difficulty} difficulty) customer ticket for "
            f"the category: '{req.category}'. The customer sentiment is {req.sentiment}. "
            f"Noise level config (0.0 to 1.0): {req.noise_level}. Reply with the ticket text only."
        )
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_prompt}],
            max_tokens=60,
            temperature=0.8
        )
        return {"category": req.category, "text": resp.choices[0].message.content.strip().strip('"')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", tags=["GenAI"])
async def analyze_sentiment(req: AnalyzeRequest):
    """Analyze the sentiment and intent of a ticket using Gen AI/LLM instead of heavy DL (Torch) pipeline."""
    if not llm_client:
        return {"sentiment": "neutral", "score": 0.5}
    try:
        sys_prompt = "You are a sentiment analyzer. Analyze this customer ticket and reply ONLY with one of: positive, neutral, negative."
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": req.text}],
            max_tokens=10,
            temperature=0.1
        )
        val = resp.choices[0].message.content.strip().lower().strip('."\'')
        s_score = 0.9 if val == "positive" else (0.1 if val == "negative" else 0.5)
        return {"sentiment": val, "score": s_score}
    except Exception:
        return {"sentiment": "neutral", "score": 0.5}

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
        obs = env.reset(
            task_id=body.task_id or "task_easy",
            ticket_index=body.ticket_index,
            custom_ticket_text=body.custom_ticket_text,
            custom_ticket_category=body.custom_ticket_category
        )
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
    
    # Self-Evolving Environment connection
    if hasattr(env, "_rolling_scores"):
        env._rolling_scores.append(result.score)
        if len(env._rolling_scores) > 10:
            env._rolling_scores.pop(0)

    # Global Metrics computation
    if "passed_episodes" not in _metrics:
        _metrics["passed_episodes"] = 0
    if result.score >= 0.7:
        _metrics["passed_episodes"] += 1
        
    tot = max(1, _metrics["total_episodes"])
    env._resolution_rate = _metrics["passed_episodes"] / tot

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

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
