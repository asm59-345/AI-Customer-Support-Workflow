"""
Pydantic models for the AI Customer Support OpenEnv Environment.
Defines typed interfaces for Observations, Actions, and Step Results.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


# ---------------------------------------------------------------------------
# Action Types
# ---------------------------------------------------------------------------

class ActionType:
    CLASSIFY  = "classify"
    RESPOND   = "respond"
    ESCALATE  = "escalate"
    CLOSE     = "close"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class TicketCategory(BaseModel):
    """Represents a ticket issue category."""
    name: str
    description: str
    keywords: List[str]


class Action(BaseModel):
    """
    An action the agent takes in the environment.
    
    Fields:
        action_type: One of 'classify', 'respond', 'escalate', 'close'
        content:     The textual content of the action (category name, reply text, etc.)
        confidence:  Optional confidence score between 0.0 and 1.0
        task_id:     Optional task identifier this action belongs to
    """
    action_type: str = Field(..., description="Type of action: classify, respond, escalate, close")
    content: str = Field(..., description="Textual content of the action")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence (0-1)")
    task_id: Optional[str] = Field(default=None, description="Task identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "classify",
                "content": "refund",
                "confidence": 0.92,
                "task_id": "task_easy"
            }
        }


class Observation(BaseModel):
    """
    The environment's observation returned after reset() or step().
    
    Fields:
        ticket:       Current customer support ticket text
        history:      List of all previous agent actions (content strings)
        status:       Current ticket status: 'open', 'responded', 'escalated', 'closed'
        step_count:   Number of steps taken so far
        task_id:      Current active task ID
        category:     Ground truth category (revealed after classification)
        hint:         Optional contextual hint for the agent
    """
    ticket: str = Field(..., description="Customer support ticket text")
    history: List[str] = Field(default_factory=list, description="Agent action history")
    status: str = Field(default="open", description="Ticket status")
    step_count: int = Field(default=0, description="Step counter")
    task_id: str = Field(default="task_easy", description="Current task ID")
    category: Optional[str] = Field(default=None, description="Revealed ticket category")
    hint: Optional[str] = Field(default=None, description="Optional hint for agent")
    difficulty_level: Optional[str] = Field(default=None, description="Difficulty of the environment")
    domain: Optional[str] = Field(default=None, description="Domain of the scenario (e.g. E-commerce)")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket": "I haven't received my order and it's been 2 weeks!",
                "history": [],
                "status": "open",
                "step_count": 0,
                "task_id": "task_medium",
                "category": None,
                "hint": "Think about whether this is a refund or shipping issue."
            }
        }


class StepResult(BaseModel):
    """
    The full result returned by step().
    
    Fields:
        observation:  New environment state
        reward:       Reward signal for the action taken
        done:         True when episode is complete
        info:         Debug/metadata dictionary
        cumulative_reward: Total reward accumulated so far
    """
    observation: Observation
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(default=False, description="Whether episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra info")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")

    class Config:
        json_schema_extra = {
            "example": {
                "observation": {},
                "reward": 0.3,
                "done": False,
                "info": {"feedback": "Correct classification!"},
                "cumulative_reward": 0.3
            }
        }


class EnvState(BaseModel):
    """Full environment state snapshot (returned by state())."""
    episode_id: str
    step_count: int
    ticket: str
    status: str
    task_id: str
    history: List[str]
    cumulative_reward: float
    started_at: float
    is_done: bool
    difficulty_level: str = "normal"
    domain: str = "general"
    resolution_rate: float = 0.0
    avg_steps_per_ticket: float = 0.0
    escalation_accuracy: float = 0.0
    sentiment_improvement: float = 0.0


class TaskDefinition(BaseModel):
    """Describes a hackathon task."""
    task_id: str
    name: str
    difficulty: str
    description: str
    allowed_actions: List[str]
    max_steps: int
    target_score: float
    reward_breakdown: Dict[str, float]


class GradeRequest(BaseModel):
    """Request body for the /grade endpoint."""
    output: str = Field(..., description="Agent's full output text to grade")
    task_id: str = Field(default="task_easy", description="Task to grade against")


class GradeResult(BaseModel):
    """Result from the /grade endpoint."""
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: List[str]
    criteria_met: Dict[str, bool]
    task_id: str


class ResetRequest(BaseModel):
    """Optional body for POST /reset"""
    task_id: Optional[str] = Field(default="task_easy")
    ticket_index: Optional[int] = Field(default=None, description="Force a specific ticket (for testing)")
    custom_ticket_text: Optional[str] = Field(default=None, description="Dynamically generated ticket text")
    custom_ticket_category: Optional[str] = Field(default=None, description="Dynamically generated ticket category")
    domain: Optional[str] = Field(default=None, description="Custom domain for the generated ticket")
    difficulty: Optional[str] = Field(default=None, description="Custom difficulty level")
    noise_level: Optional[float] = Field(default=0.0, description="Noise level to inject")
    sentiment: Optional[str] = Field(default=None, description="Custom sentiment")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    version: str
    environment: str
