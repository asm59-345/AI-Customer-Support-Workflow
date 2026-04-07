"""
Deterministic grader for the AI Customer Support OpenEnv Environment.

Handles all 10 workflow types:
  inquiry_management, refund_returns, technical_support, billing_payments,
  account_access, order_fulfillment, complaint_escalation,
  appointment_scheduling, feedback_survey, proactive_engagement
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from app.models import GradeResult
from app.tasks import CATEGORY_SYNONYMS, TICKET_CORPUS


# ---------------------------------------------------------------------------
# Response Quality Keywords
# ---------------------------------------------------------------------------

EMPATHY_PHRASES = [
    "sorry", "apologize", "apologies", "understand", "sincerely",
    "unfortunately", "regret", "deeply", "appreciate", "truly",
    "completely understand", "can imagine", "frustrating",
]

RESOLUTION_PHRASES = [
    "refund", "replacement", "fix", "resolve", "solution", "help",
    "process", "escalate", "investigate", "update", "within",
    "business days", "team", "immediately", "arrange", "schedule",
    "assign", "contact", "notify", "send", "initiate", "confirm",
]

PROFESSIONALISM_PHRASES = [
    "please", "thank you", "let us know", "feel free",
    "happy to", "assist", "support", "available", "our team",
    "right away", "promptly",
]

PERSONALIZATION_PHRASES = [
    "your account", "your order", "your case", "your ticket",
    "for you", "specifically", "personally", "dedicated",
]


# ---------------------------------------------------------------------------
# Workflow-specific response validators
# ---------------------------------------------------------------------------

# Keys must match ticket["category"] values from tasks.py exactly
WORKFLOW_VALIDATORS: Dict[str, List[str]] = {
    "inquiry":              ["information", "happy to", "let us know", "help", "answer", "available", "promptly"],
    "refund":               ["refund", "return", "replacement", "business days", "process", "initiat"],
    "technical_issue":      ["engineer", "update", "version", "team", "technical", "fix", "resolve", "cache"],
    "billing":              ["charge", "refund", "billing", "payment", "account", "erroneous", "invoice"],
    "account":              ["account", "password", "reset", "access", "verify", "security", "forgot"],
    "shipping":             ["order", "shipping", "delivery", "tracking", "courier", "investigation", "replacement"],
    "complaint":            ["escalat", "senior", "manager", "immediately", "priority", "sorry", "standard"],
    "appointment":          ["schedule", "appointment", "time", "slot", "calendar", "available", "invite"],
    "feedback":             ["thank you", "feedback", "appreciate", "improve", "noted", "insights"],
    "proactive_engagement": ["offer", "discount", "exclusive", "loyalty", "upgrade", "valued", "tailored"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip()


def _check_classification(predicted: str, ground_truth: str) -> Tuple[float, str]:
    """Score classification with synonym flexibility."""
    predicted_n = _normalize(predicted)
    gt_n = _normalize(ground_truth)

    if predicted_n == gt_n:
        return 0.7, f"✅ Perfect classification: '{ground_truth}'"

    synonyms = CATEGORY_SYNONYMS.get(ground_truth, [])
    for syn in synonyms:
        if syn in predicted_n or predicted_n in syn:
            return 0.5, f"⚠️ Partial match via synonym: '{predicted_n}' ≈ '{ground_truth}'"

    gt_words = set(gt_n.replace("_", " ").split())
    pred_words = set(predicted_n.replace("_", " ").split())
    if gt_words & pred_words:
        return 0.3, f"⚠️ Weak match — keyword overlap with '{ground_truth}'"

    return -0.2, f"❌ Wrong classification: '{predicted_n}' (expected '{ground_truth}')"


def _score_response_quality(response_text: str, ticket: Dict) -> Tuple[float, List[str]]:
    """Score quality of agent's respond action across all 10 workflow types."""
    text = _normalize(response_text)
    feedback = []
    score = 0.0
    workflow = ticket.get("category", "")

    # Empathy check
    empathy_found = [p for p in EMPATHY_PHRASES if p in text]
    if empathy_found:
        score += 0.15
        feedback.append(f"✅ Empathy detected ({', '.join(empathy_found[:2])})")
    else:
        feedback.append("⚠️ No empathy — acknowledge the customer's experience first")

    # Resolution check
    resolution_found = [p for p in RESOLUTION_PHRASES if p in text]
    if resolution_found:
        score += 0.15
        feedback.append(f"✅ Resolution offered ({', '.join(resolution_found[:2])})")
    else:
        feedback.append("⚠️ No resolution — provide a concrete next step or action")

    # Workflow-specific vocabulary
    workflow_terms = WORKFLOW_VALIDATORS.get(workflow, [])
    wf_found = [t for t in workflow_terms if t in text]
    if wf_found:
        score += 0.10
        feedback.append(f"✅ Workflow-relevant terms ({', '.join(wf_found[:2])})")
    else:
        feedback.append(f"⚠️ Missing {workflow} workflow context in response")

    # Professionalism
    prof_found = [p for p in PROFESSIONALISM_PHRASES if p in text]
    if prof_found:
        score += 0.05
        feedback.append("✅ Professional tone")

    # Personalization
    personal_found = [p for p in PERSONALIZATION_PHRASES if p in text]
    if personal_found:
        score += 0.025
        feedback.append("✅ Personalized response")

    # Keyword relevance
    ticket_keywords = ticket.get("keywords", [])
    matched = [kw for kw in ticket_keywords if _normalize(kw) in text]
    if matched:
        score += 0.025 * min(len(matched), 3)
        feedback.append(f"✅ Key issue addressed: {matched[:3]}")
    else:
        feedback.append("⚠️ Response doesn't reference specific ticket details")

    # Length
    word_count = len(response_text.split())
    if word_count < 8:
        score -= 0.1
        feedback.append("❌ Response too short (< 8 words)")
    elif word_count >= 20:
        score += 0.025
        feedback.append("✅ Adequately detailed response")

    return round(min(max(score, 0.0), 0.5), 3), feedback


def _check_escalation(did_escalate: bool, should_escalate: bool) -> Tuple[float, str]:
    """Score escalation decision correctness."""
    if did_escalate and should_escalate:
        return 0.3, "✅ Correct escalation — complex/angry ticket warranted senior support"
    elif not did_escalate and not should_escalate:
        return 0.1, "✅ Correct no-escalation — issue was resolvable at first-contact"
    elif did_escalate and not should_escalate:
        return -0.2, "❌ False escalation — this routine issue didn't need senior support"
    else:
        return -0.3, "❌ Missed escalation — high-severity ticket required escalation"


# ---------------------------------------------------------------------------
# Grade Step (live, per-action)
# ---------------------------------------------------------------------------

def grade_step_action(action_type: str, content: str, ticket: Dict,
                      task_id: str, history: List[str]) -> Tuple[float, str, bool]:
    """Grade a single step action during an episode."""
    reward = 0.0
    feedback = ""
    done = False

    if action_type == "classify":
        score, fb = _check_classification(content, ticket["category"])
        reward, feedback = score, fb

    elif action_type == "respond":
        if task_id == "task_easy":
            reward = -0.1
            feedback = "❌ 'respond' not allowed in task_easy"
        else:
            score, fbs = _score_response_quality(content, ticket)
            reward, feedback = score, " | ".join(fbs)

    elif action_type == "escalate":
        if task_id in ("task_easy", "task_medium"):
            reward = -0.1
            feedback = f"❌ 'escalate' not allowed in {task_id}"
        else:
            score, fb = _check_escalation(True, ticket["should_escalate"])
            reward, feedback = score, fb

    elif action_type == "close":
        if len(history) >= 2:
            reward = 0.1
            feedback = "✅ Ticket closed cleanly"
            done = True
        else:
            reward = -0.1
            feedback = "⚠️ Too early to close — complete classify + respond first"

    else:
        reward = -0.2
        feedback = f"❌ Unknown action type '{action_type}'"

    return round(reward, 3), feedback, done


# ---------------------------------------------------------------------------
# Grade Full Output (for /grade endpoint)
# ---------------------------------------------------------------------------

def grade_output(output: str, task_id: str, ticket_index: int = 0) -> GradeResult:
    """
    Grade the complete agent output against the task criteria.

    Weight design ensures max achievable score >= target for each task:
      task_easy:   cls(max 0.7) * 1.0                          -> max 0.70  (target 0.70)
      task_medium: cls(max 0.7) * 0.5 + rsp(max 0.5) * 1.0    -> max 0.85  (target 0.75)
      task_hard:   cls * 0.5   + rsp * 1.0 + esc(max 0.3)*0.5 -> max 1.00  (target 0.80)

    IMPORTANT: passes only the response text (not the classification prefix)
    to _score_response_quality so keyword matching is accurate.
    """
    text = _normalize(output)
    ticket = TICKET_CORPUS[ticket_index % len(TICKET_CORPUS)]
    feedback: List[str] = []
    criteria_met: Dict[str, bool] = {}
    total_score = 0.0

    # Split output: first word = classification, rest = response text
    parts = output.split(" ", 1)
    classification_part = parts[0].strip()
    response_part = parts[1].strip() if len(parts) > 1 else output

    if task_id == "task_easy":
        cls_score, cls_fb = _check_classification(classification_part, ticket["category"])
        total_score = max(0.0, cls_score)
        feedback.append(cls_fb)
        criteria_met["correct_classification"] = cls_score > 0.4

    elif task_id == "task_medium":
        cls_score, cls_fb = _check_classification(classification_part, ticket["category"])
        rsp_score, rsp_fbs = _score_response_quality(response_part, ticket)
        total_score = max(0.0, cls_score * 0.5 + rsp_score * 1.0)
        feedback.append(cls_fb)
        feedback.extend(rsp_fbs)
        criteria_met["correct_classification"] = cls_score > 0.4
        criteria_met["quality_response"] = rsp_score > 0.2

    elif task_id == "task_hard":
        cls_score, cls_fb = _check_classification(classification_part, ticket["category"])
        rsp_score, rsp_fbs = _score_response_quality(response_part, ticket)
        did_esc = "escalat" in text
        esc_score, esc_fb = _check_escalation(did_esc, ticket["should_escalate"])
        total_score = max(0.0, cls_score * 0.5 + rsp_score * 1.0 + esc_score * 0.5)
        feedback.append(cls_fb)
        feedback.extend(rsp_fbs)
        feedback.append(esc_fb)
        criteria_met["correct_classification"] = cls_score > 0.4
        criteria_met["quality_response"] = rsp_score > 0.2
        criteria_met["correct_escalation"] = esc_score > 0.0

    return GradeResult(
        score=round(min(total_score, 1.0), 3),
        feedback=feedback,
        criteria_met=criteria_met,
        task_id=task_id,
    )
