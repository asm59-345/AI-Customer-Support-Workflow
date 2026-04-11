"""
Core OpenEnv Environment: SupportEnv

Implements the standard OpenEnv interface:
  - reset()  → Observation
  - step()   → StepResult (observation, reward, done, info)
  - state()  → EnvState
"""

from __future__ import annotations
import uuid
import time
import random
from typing import Optional

from app.models import (
    Action, Observation, StepResult, EnvState, ActionType
)
from app.tasks import TICKET_CORPUS, TASKS, get_task
from app.grader import grade_step_action


class SupportEnv:
    """
    AI Customer Support Reinforcement Learning Environment.

    Implements the OpenEnv standard interface (reset / step / state)
    compatible with gymnasium-style RL frameworks.

    Episode Flow:
        1. reset(task_id)     → receive initial observation (ticket)
        2. step(action)       → execute action, receive reward + obs
        3. Repeat until done  → 'done' flag triggers episode end
        4. state()            → inspect current episode metadata

    Reward Range: [-0.3, +0.7] per step; cumulative episode target: ≥ 0.7
    """

    VERSION = "1.0.0"
    ENV_NAME = "ai-customer-support-v1"

    def __init__(self):
        self._episode_id: Optional[str] = None
        self._ticket: Optional[dict] = None
        self._task_id: str = "task_easy"
        self._history: list = []
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._status: str = "open"
        self._is_done: bool = False
        self._started_at: float = 0.0
        self._classified: bool = False
        self._responded: bool = False
        self._escalated: bool = False

        # Self-Evolving Trackers
        self._rolling_scores: list = []
        self._current_difficulty: str = "normal"
        self._current_noise: float = 0.0
        self._current_domain: str = "E-commerce"
        self._resolution_rate: float = 0.0

    # -----------------------------------------------------------------------
    # OpenEnv Core API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy", ticket_index: Optional[int] = None, custom_ticket_text: Optional[str] = None, custom_ticket_category: Optional[str] = None) -> Observation:
        """
        Initialize a new episode.

        Args:
            task_id:       Which task to run (task_easy | task_medium | task_hard)
            ticket_index:  Force a specific ticket index (default: random)
            custom_ticket_text: Custom Gen AI ticket text
            custom_ticket_category: Custom Gen AI ticket category

        Returns:
            Initial Observation with ticket text and empty history.
        """
        task = get_task(task_id)  # validates task_id

        # Self-Evolving: adjust difficulty based on performance history
        if len(self._rolling_scores) >= 3:
            avg_perf = sum(self._rolling_scores[-3:]) / 3
            if avg_perf >= 0.8:
                self._current_difficulty = "hard"
                self._current_noise = 0.2
            elif avg_perf >= 0.6:
                self._current_difficulty = "medium"
                self._current_noise = 0.1
            else:
                self._current_difficulty = "normal"
                self._current_noise = 0.0

        doms = ["E-commerce", "Banking", "SaaS", "Telecom"]
        self._current_domain = random.choice(doms)

        # Pick ticket
        if custom_ticket_text and custom_ticket_category:
            self._ticket = {
                "text": custom_ticket_text,
                "category": custom_ticket_category,
                "workflow_type": "custom_gen_ai",
                "sentiment": "neutral",
                "should_escalate": False,
                "keywords": []
            }
            self._ticket_index = -1
        elif ticket_index is not None:
            idx = ticket_index % len(TICKET_CORPUS)
            self._ticket = TICKET_CORPUS[idx]
            self._ticket_index = idx  # save for grader
        else:
            idx = random.randint(0, len(TICKET_CORPUS) - 1)
            self._ticket = TICKET_CORPUS[idx]
            self._ticket_index = idx  # save for grader

        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())
        self._history = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._status = "open"
        self._is_done = False
        self._started_at = time.time()
        self._classified = False
        self._responded = False
        self._escalated = False

        return Observation(
            ticket=self._ticket["text"],
            history=[],
            status="open",
            step_count=0,
            task_id=task_id,
            category=None,
            hint=self._build_hint(task_id),
            difficulty_level=self._current_difficulty,
            domain=self._current_domain
        )

    def step(self, action: Action) -> StepResult:
        """
        Execute an agent action in the environment.

        Args:
            action: An Action object with action_type and content.

        Returns:
            StepResult containing new observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: If called before reset().
        """
        if self._ticket is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._is_done:
            return StepResult(
                observation=self._current_observation(),
                reward=0.0,
                done=True,
                info={"warning": "Episode already completed. Call reset() to start a new one."},
                cumulative_reward=self._cumulative_reward,
            )

        self._step_count += 1
        task = get_task(self._task_id)

        # Validate action type
        if action.action_type not in task.allowed_actions:
            reward = -0.15
            self._cumulative_reward += reward
            feedback = f"❌ Action '{action.action_type}' not allowed in {self._task_id}. Allowed: {task.allowed_actions}"
            return StepResult(
                observation=self._current_observation(),
                reward=reward,
                done=False,
                info={"feedback": feedback, "step": self._step_count},
                cumulative_reward=round(self._cumulative_reward, 3),
            )

        # Grade step
        reward, feedback, step_done = grade_step_action(
            action.action_type, action.content,
            self._ticket, self._task_id, self._history
        )

        # Update state flags
        if action.action_type == ActionType.CLASSIFY:
            self._classified = True
            self._status = "classified"
        elif action.action_type == ActionType.RESPOND:
            self._responded = True
            self._status = "responded"
        elif action.action_type == ActionType.ESCALATE:
            self._escalated = True
            self._status = "escalated"
        elif action.action_type == ActionType.CLOSE:
            self._status = "closed"
            step_done = True

        # Add to history
        self._history.append(f"[{action.action_type.upper()}] {action.content[:120]}")
        self._cumulative_reward += reward

        # Check episode termination
        done = step_done or self._check_done()

        if done:
            self._is_done = True
            self._status = "closed" if self._status == "responded" else self._status

        obs = self._current_observation()

        return StepResult(
            observation=obs,
            reward=round(reward, 3),
            done=done,
            info={
                "feedback": feedback,
                "step": self._step_count,
                "episode_id": self._episode_id,
                "classified": self._classified,
                "responded": self._responded,
                "escalated": self._escalated,
                "ticket_category": self._ticket["category"] if (done or self._classified) else "hidden",
            },
            cumulative_reward=round(self._cumulative_reward, 3),
        )

    def state(self) -> EnvState:
        """
        Return the current episode state metadata.

        Returns:
            EnvState snapshot of the current episode.
        """
        return EnvState(
            episode_id=self._episode_id or "not_started",
            step_count=self._step_count,
            ticket=self._ticket["text"] if self._ticket else "",
            status=self._status,
            task_id=self._task_id,
            history=self._history.copy(),
            cumulative_reward=round(self._cumulative_reward, 3),
            started_at=self._started_at,
            is_done=self._is_done,
            difficulty_level=self._current_difficulty,
            domain=self._current_domain,
            resolution_rate=self._resolution_rate,
            avg_steps_per_ticket=self._step_count if self._is_done else 0.0,
            escalation_accuracy=1.0 if self._escalated else 0.0,
            sentiment_improvement=0.5
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _current_observation(self) -> Observation:
        """Build a current Observation from environment state."""
        return Observation(
            ticket=self._ticket["text"] if self._ticket else "",
            history=self._history.copy(),
            status=self._status,
            step_count=self._step_count,
            task_id=self._task_id,
            category=self._ticket["category"] if self._classified else None,
            hint=None,
            difficulty_level=self._current_difficulty,
            domain=self._current_domain
        )

    def _check_done(self) -> bool:
        """Determine if episode should end."""
        task = TASKS.get(self._task_id)
        if task and self._step_count >= task.max_steps:
            return True

        if self._task_id == "task_easy" and self._classified:
            return True

        if self._task_id == "task_medium" and self._classified and self._responded:
            return True

        if self._task_id == "task_hard":
            # Done if: classified + responded + (escalated or explicitly closed)
            if self._classified and self._responded and (self._escalated or self._step_count >= 4):
                return True

        return False

    def _build_hint(self, task_id: str) -> str:
        """Build a contextual hint for the agent based on task."""
        hints = {
            "task_easy": "Read the ticket carefully and choose the most fitting category.",
            "task_medium": "First classify the ticket, then craft an empathetic, solution-focused response.",
            "task_hard": "Follow the full workflow: classify → respond → escalate (if the issue is severe).",
        }
        return hints.get(task_id, "Complete the task.")
