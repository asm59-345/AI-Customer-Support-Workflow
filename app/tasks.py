"""
Expanded Task & Ticket definitions for the AI Customer Support OpenEnv Environment.

Workflow Types (inspired by real-world CX platforms):
  1. Inquiry Management        — general questions, product info, policy
  2. Refund & Returns          — money back, return requests
  3. Technical Support         — bugs, crashes, errors
  4. Billing & Payments        — charges, invoices, subscriptions
  5. Account & Access          — login, password, profile
  6. Order & Fulfillment       — shipping, delivery, tracking
  7. Complaint & Escalation    — angry customers, SLA breach
  8. Appointment Scheduling    — booking, rescheduling
  9. Feedback & Survey         — ratings, reviews, NPS
 10. Proactive Engagement      — upsell, onboarding, retention
"""

from __future__ import annotations
from typing import Dict, List
from app.models import TaskDefinition


# ---------------------------------------------------------------------------
# Ticket Corpus — 30 diverse, realistic support tickets across 10 workflow types
# ---------------------------------------------------------------------------

TICKET_CORPUS: List[Dict] = [

    # ── 1. Inquiry Management ───────────────────────────────────────────────
    {
        "text": "Do you offer student discounts? I'm a university student and would love to try your premium plan.",
        "category": "inquiry",
        "workflow_type": "inquiry_management",
        "sentiment": "positive",
        "should_escalate": False,
        "keywords": ["student", "discount", "university", "premium", "plan"],
        "ideal_response": "Great news! We offer a 30% student discount. Verify your student status at our Student Hub and the discount applies automatically.",
    },
    {
        "text": "What are your business hours and do you offer 24/7 support?",
        "category": "inquiry",
        "workflow_type": "inquiry_management",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["hours", "support", "24/7", "available"],
        "ideal_response": "Our AI support is available 24/7. Human agents are available Monday–Friday 9AM–6PM EST. We're always happy to help!",
    },
    {
        "text": "Can you explain the difference between your Basic and Pro subscription plans?",
        "category": "inquiry",
        "workflow_type": "inquiry_management",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["basic", "pro", "subscription", "plan", "difference"],
        "ideal_response": "The Basic plan includes core features for individuals. The Pro plan adds advanced analytics, priority support, and team collaboration tools. Happy to compare them in detail!",
    },

    # ── 2. Refund & Returns ─────────────────────────────────────────────────
    {
        "text": "I ordered a product 2 weeks ago and still haven't received it. I want a full refund immediately!",
        "category": "refund",
        "workflow_type": "refund_returns",
        "sentiment": "angry",
        "should_escalate": False,
        "keywords": ["refund", "order", "received", "full refund", "immediately"],
        "ideal_response": "We sincerely apologize for the delay. We have initiated a full refund to your account. You should see it within 3-5 business days.",
    },
    {
        "text": "I received a damaged item. How do I return it and get a replacement?",
        "category": "refund",
        "workflow_type": "refund_returns",
        "sentiment": "upset",
        "should_escalate": False,
        "keywords": ["damaged", "return", "replacement", "item"],
        "ideal_response": "We're so sorry about the damaged item! Please share a photo and we'll arrange a free return pickup and ship a replacement within 2 business days.",
    },
    {
        "text": "I changed my mind about the purchase. It's been 5 days. Can I still return it for a full refund?",
        "category": "refund",
        "workflow_type": "refund_returns",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["return", "refund", "purchase", "5 days"],
        "ideal_response": "Yes! Our return policy allows returns within 30 days of purchase. Simply initiate the return from your Orders page and we'll process the refund within 5-7 business days.",
    },

    # ── 3. Technical Support ────────────────────────────────────────────────
    {
        "text": "My laptop keeps crashing every time I open the app. The screen goes black and I have to restart.",
        "category": "technical_issue",
        "workflow_type": "technical_support",
        "sentiment": "frustrated",
        "should_escalate": True,
        "keywords": ["crashing", "screen", "restart", "app", "black"],
        "ideal_response": "We're sorry about the technical issue. Please update the app to version 3.2.1 and clear the cache. Our engineers have been notified and will follow up within 24 hours.",
    },
    {
        "text": "The API integration is returning a 503 error after the latest update. Our entire workflow is broken.",
        "category": "technical_issue",
        "workflow_type": "technical_support",
        "sentiment": "alarmed",
        "should_escalate": True,
        "keywords": ["API", "503", "error", "integration", "broken", "update"],
        "ideal_response": "We sincerely apologize for the service disruption. This is a known issue with v3.2 and our engineering team is working on a hotfix. Expected resolution: 2 hours. We'll notify you immediately.",
    },
    {
        "text": "Your mobile app is extremely slow and takes over 30 seconds to load. This is unusable.",
        "category": "technical_issue",
        "workflow_type": "technical_support",
        "sentiment": "frustrated",
        "should_escalate": False,
        "keywords": ["slow", "load", "mobile", "app", "unusable"],
        "ideal_response": "We apologize for the poor performance. We've identified load issues on iOS 17+ and have a fix in review. Try force-quitting and relaunching. Estimated update: 48 hours.",
    },

    # ── 4. Billing & Payments ───────────────────────────────────────────────
    {
        "text": "I was charged twice for my subscription this month. Please fix this billing error.",
        "category": "billing",
        "workflow_type": "billing_payments",
        "sentiment": "upset",
        "should_escalate": False,
        "keywords": ["charged", "twice", "subscription", "billing", "fix"],
        "ideal_response": "We apologize for the duplicate billing. We've identified the error and will refund the extra charge within 2-3 business days. Thank you for bringing this to our attention.",
    },
    {
        "text": "My payment was declined but the money was still deducted from my bank account!",
        "category": "billing",
        "workflow_type": "billing_payments",
        "sentiment": "alarmed",
        "should_escalate": True,
        "keywords": ["payment", "declined", "deducted", "bank"],
        "ideal_response": "We apologize for this issue. The declined payment may show as a temporary bank hold that should release within 48 hours. If not resolved, please contact us with your transaction ID.",
    },
    {
        "text": "How do I upgrade to the annual plan and get the 20% discount? I can't find the option.",
        "category": "billing",
        "workflow_type": "billing_payments",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["upgrade", "annual", "discount", "plan"],
        "ideal_response": "To upgrade, go to Settings > Billing > Change Plan > Annual. The 20% discount applies automatically at checkout. Would you like me to initiate this for you?",
    },

    # ── 5. Account & Access ─────────────────────────────────────────────────
    {
        "text": "Hi, I can't seem to log into my account. It says my password is wrong but I'm sure it's correct.",
        "category": "account",
        "workflow_type": "account_access",
        "sentiment": "confused",
        "should_escalate": False,
        "keywords": ["log in", "account", "password", "wrong"],
        "ideal_response": "Sorry for the trouble! Please use 'Forgot Password' to reset your credentials. If you still can't access your account, our team can verify your identity and restore access.",
    },
    {
        "text": "I think my account has been hacked. There are login attempts from a country I've never been to.",
        "category": "account",
        "workflow_type": "account_access",
        "sentiment": "alarmed",
        "should_escalate": True,
        "keywords": ["hacked", "login", "country", "security", "unauthorized"],
        "ideal_response": "We take security very seriously. We've temporarily locked your account to protect you. Please verify your identity via email and reset your password immediately. Our security team is reviewing this.",
    },
    {
        "text": "I'd like to change the email address associated with my account. How do I do that?",
        "category": "account",
        "workflow_type": "account_access",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["change", "email", "account"],
        "ideal_response": "Go to Settings > Profile > Edit Email Address. You'll receive a verification link at your new address to confirm the change. Let us know if you need help!",
    },

    # ── 6. Order & Fulfillment ──────────────────────────────────────────────
    {
        "text": "Where is my package? The tracking number shows it was delivered last Thursday but I never got it.",
        "category": "shipping",
        "workflow_type": "order_fulfillment",
        "sentiment": "worried",
        "should_escalate": True,
        "keywords": ["package", "tracking", "delivered", "never got"],
        "ideal_response": "We're sorry your package hasn't arrived. We've opened an investigation with the courier. You'll hear from us within 24 hours, and we'll send a replacement if it's confirmed lost.",
    },
    {
        "text": "I need to change the delivery address on my order. It hasn't shipped yet.",
        "category": "shipping",
        "workflow_type": "order_fulfillment",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["delivery", "address", "order", "change", "shipped"],
        "ideal_response": "We can update the delivery address for you. Please provide your order number and the new address, and we'll make the change right away before it ships!",
    },
    {
        "text": "My order arrived but it's completely wrong — I ordered a blue version and got red.",
        "category": "shipping",
        "workflow_type": "order_fulfillment",
        "sentiment": "upset",
        "should_escalate": False,
        "keywords": ["order", "wrong", "blue", "red", "incorrect"],
        "ideal_response": "We sincerely apologize for the mix-up! We'll send the correct item with expedited shipping at no charge. A prepaid return label for the incorrect item will be emailed to you shortly.",
    },

    # ── 7. Complaint & Escalation ───────────────────────────────────────────
    {
        "text": "This is absolutely unacceptable! Your product broke within a week of purchase and your support is non-existent!",
        "category": "complaint",
        "workflow_type": "complaint_escalation",
        "sentiment": "very_angry",
        "should_escalate": True,
        "keywords": ["unacceptable", "broke", "purchase", "support"],
        "ideal_response": "We are deeply sorry for your experience. This is not our standard. I'm escalating your case to our senior support team immediately and a manager will contact you within 2 hours.",
    },
    {
        "text": "I've submitted 5 tickets over 2 weeks and nobody has responded. This is a disgrace!",
        "category": "complaint",
        "workflow_type": "complaint_escalation",
        "sentiment": "furious",
        "should_escalate": True,
        "keywords": ["tickets", "weeks", "responded", "disgrace", "5"],
        "ideal_response": "We are truly sorry for this unacceptable lapse in service. I'm personally escalating all your cases to our head of support. You will receive a call from a senior manager within the hour.",
    },
    {
        "text": "Your chatbot gave me completely wrong information and now I've lost $200 because of it.",
        "category": "complaint",
        "workflow_type": "complaint_escalation",
        "sentiment": "very_angry",
        "should_escalate": True,
        "keywords": ["chatbot", "wrong", "information", "lost", "200"],
        "ideal_response": "We are deeply sorry. This should never have happened. I'm escalating this immediately to our compliance team. We will investigate fully and make it right — please expect a callback within 30 minutes.",
    },

    # ── 8. Appointment Scheduling ───────────────────────────────────────────
    {
        "text": "I need to schedule a product demo with your sales team. What time slots are available this week?",
        "category": "appointment",
        "workflow_type": "appointment_scheduling",
        "sentiment": "positive",
        "should_escalate": False,
        "keywords": ["schedule", "demo", "sales", "time", "week"],
        "ideal_response": "We'd love to show you a demo! Available slots this week: Wednesday 2PM, Thursday 10AM, Friday 3PM EST. Which works best for you? I'll send a calendar invite right away!",
    },
    {
        "text": "I need to reschedule my support call from tomorrow. Can we move it to next Monday?",
        "category": "appointment",
        "workflow_type": "appointment_scheduling",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["reschedule", "call", "tomorrow", "Monday"],
        "ideal_response": "Of course! I've rescheduled your support call to next Monday. You'll receive a new calendar invitation at your registered email address. Is 10AM Monday EST suitable?",
    },

    # ── 9. Feedback & Survey ────────────────────────────────────────────────
    {
        "text": "I just wanted to say your support agent Sarah was absolutely amazing. She solved my issue in minutes!",
        "category": "feedback",
        "workflow_type": "feedback_survey",
        "sentiment": "very_positive",
        "should_escalate": False,
        "keywords": ["support", "agent", "amazing", "solved", "minutes"],
        "ideal_response": "Thank you so much for the wonderful feedback! We're thrilled that Sarah was able to help. We'll make sure she's recognized for her excellent service. Is there anything else we can do for you?",
    },
    {
        "text": "I gave you 2 stars in the app survey. The product is fine but the onboarding was confusing and took too long.",
        "category": "feedback",
        "workflow_type": "feedback_survey",
        "sentiment": "negative",
        "should_escalate": False,
        "keywords": ["stars", "survey", "onboarding", "confusing", "long"],
        "ideal_response": "Thank you for the honest feedback! We're sorry the onboarding fell short. We're actively improving it based on exactly this input. May we schedule a 15-min walkthrough to get you up to speed?",
    },

    # ── 10. Proactive Engagement ────────────────────────────────────────────
    {
        "text": "I've been a customer for 3 years and my contract is up soon. What's the best renewal deal you can offer me?",
        "category": "proactive_engagement",
        "workflow_type": "proactive_engagement",
        "sentiment": "neutral",
        "should_escalate": False,
        "keywords": ["customer", "years", "contract", "renewal", "deal"],
        "ideal_response": "Thank you for your loyalty! As a valued 3-year customer, we're offering you an exclusive 25% discount on annual renewal plus a free upgrade to Pro. Shall I lock that in for you today?",
    },
    {
        "text": "I just signed up but I have no idea where to start. The platform is overwhelming.",
        "category": "proactive_engagement",
        "workflow_type": "proactive_engagement",
        "sentiment": "confused",
        "should_escalate": False,
        "keywords": ["signed up", "start", "platform", "overwhelming", "new"],
        "ideal_response": "Welcome aboard! Don't worry — we've got you covered. I'm assigning you a dedicated onboarding specialist and sending a personalized quickstart guide to your email right now!",
    },
]


# ---------------------------------------------------------------------------
# Category Synonyms (used by grader for flexible matching)
# ---------------------------------------------------------------------------

CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "inquiry":             ["inquiry", "question", "general", "info", "how to", "what is", "policy"],
    "refund":              ["refund", "return", "money back", "reimbursement", "refunded"],
    "technical_issue":     ["technical", "tech issue", "bug", "crash", "error", "broken", "not working", "slow"],
    "billing":             ["billing", "charge", "payment", "invoice", "charged", "fee", "subscription"],
    "account":             ["account", "login", "password", "profile", "access", "security"],
    "shipping":            ["shipping", "delivery", "package", "parcel", "tracking", "order", "fulfillment"],
    "complaint":           ["complaint", "angry", "unhappy", "unacceptable", "escalate", "disgrace"],
    "appointment":         ["appointment", "schedule", "demo", "booking", "reschedule", "call"],
    "feedback":            ["feedback", "review", "rating", "survey", "nps", "satisfaction"],
    "proactive_engagement": ["engagement", "retention", "upsell", "renewal", "onboarding", "loyalty"],
}


# ---------------------------------------------------------------------------
# Workflow Type Labels (for display / analytics)
# ---------------------------------------------------------------------------

WORKFLOW_TYPES: Dict[str, str] = {
    "inquiry_management":    "📋 Inquiry Management",
    "refund_returns":        "💰 Refund & Returns",
    "technical_support":     "🔧 Technical Support",
    "billing_payments":      "🧾 Billing & Payments",
    "account_access":        "🔐 Account & Access",
    "order_fulfillment":     "📦 Order & Fulfillment",
    "complaint_escalation":  "🚨 Complaint & Escalation",
    "appointment_scheduling":"📅 Appointment Scheduling",
    "feedback_survey":       "⭐ Feedback & Survey",
    "proactive_engagement":  "🎯 Proactive Engagement",
}


# ---------------------------------------------------------------------------
# Task Definitions (3 difficulty levels)
# ---------------------------------------------------------------------------

TASKS: Dict[str, TaskDefinition] = {
    "task_easy": TaskDefinition(
        task_id="task_easy",
        name="🟢 Task 1: Ticket Classification",
        difficulty="Easy",
        description=(
            "Classify the customer support ticket into one of the following workflow categories: "
            "inquiry, refund, technical_issue, billing, account, shipping, complaint, "
            "appointment, feedback, proactive_engagement. "
            "Use the 'classify' action with the exact category name as content."
        ),
        allowed_actions=["classify"],
        max_steps=3,
        target_score=0.7,
        reward_breakdown={
            "correct_classification": 0.7,
            "partial_classification": 0.3,
            "wrong_classification": -0.2,
        },
    ),

    "task_medium": TaskDefinition(
        task_id="task_medium",
        name="🟡 Task 2: Classify & Respond",
        difficulty="Medium",
        description=(
            "Step 1 — Classify the ticket (use 'classify' action with the category name). "
            "Step 2 — Generate an appropriate, empathetic, resolution-focused response (use 'respond' action). "
            "The response must acknowledge the issue and provide a clear next step."
        ),
        allowed_actions=["classify", "respond"],
        max_steps=5,
        target_score=0.75,
        reward_breakdown={
            "correct_classification": 0.3,
            "quality_response": 0.5,
            "empathy_bonus": 0.1,
            "wrong_action": -0.2,
        },
    ),

    "task_hard": TaskDefinition(
        task_id="task_hard",
        name="🔴 Task 3: Full CX Workflow",
        difficulty="Hard",
        description=(
            "Execute the complete customer experience workflow: "
            "1) Classify the ticket into its workflow type, "
            "2) Generate a personalized, resolution-focused response, "
            "3) Escalate to senior support if the issue is severe, urgent, or the customer is very angry, "
            "4) Close the ticket with a confirmation message. "
            "Use actions: 'classify', 'respond', 'escalate', 'close' in the correct workflow order."
        ),
        allowed_actions=["classify", "respond", "escalate", "close"],
        max_steps=8,
        target_score=0.8,
        reward_breakdown={
            "correct_classification": 0.2,
            "quality_response": 0.3,
            "correct_escalation": 0.3,
            "missing_escalation": -0.3,
            "false_escalation": -0.2,
            "smooth_close": 0.1,
        },
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[TaskDefinition]:
    return list(TASKS.values())
