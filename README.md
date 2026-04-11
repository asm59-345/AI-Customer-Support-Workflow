---
title: AI Customer Support Environment
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🤖 AI Customer Support Environment — OpenEnv RL

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-6366f1?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge)](https://fastapi.tiangolo.com)

> A production-ready **self-evolving reinforcement learning environment** where AI agents learn to handle real-world customer support tickets across **10 CX workflow types**. Featuring a **DistilBERT Critic Agent** and multi-dimensional reward logic for robust policy learning.

---

## 🎯 Overview

This project implements the **OpenEnv standard API** (`reset()` / `step()` / `state()`) as a containerized FastAPI service deployed on Hugging Face Spaces. AI agents interact with the environment to:

1. **Classify** — Identify the ticket's workflow type
2. **Respond** — Generate empathetic, resolution-focused replies  
3. **Escalate** — Route complex cases to senior support
4. **Close** — Complete the ticket lifecycle

---

## 🗂 10 CX Workflow Types

| Icon | Type | Category Key | Escalates |
|------|------|-------------|-----------|
| 📋 | Inquiry Management | `inquiry` | No |
| 💰 | Refund & Returns | `refund` | No |
| 🔧 | Technical Support | `technical_issue` | Yes |
| 🧾 | Billing & Payments | `billing` | Yes |
| 🔐 | Account & Access | `account` | Yes |
| 📦 | Order & Fulfillment | `shipping` | Yes |
| 🚨 | Complaint & Escalation | `complaint` | Yes |
| 📅 | Appointment Scheduling | `appointment` | No |
| ⭐ | Feedback & Survey | `feedback` | No |
| 🎯 | Proactive Engagement | `proactive_engagement` | No |

---

## 🧪 Tasks

| Task | Difficulty | Actions | Max Steps | Target Score |
|------|-----------|---------|-----------|-------------|
| `task_easy` | 🟢 Easy | classify | 3 | ≥ 0.70 |
| `task_medium` | 🟡 Medium | classify + respond | 5 | ≥ 0.75 |
| `task_hard` | 🔴 Hard | classify + respond + escalate + close | 8 | ≥ 0.80 |

---

## 🔌 API Reference

### `POST /reset`
Initialize a new episode.
```json
{"task_id": "task_easy"}
```
Returns: Initial `Observation` with ticket text, status, hint.

### `POST /step`
Execute an agent action.
```json
{"action_type": "classify", "content": "billing", "confidence": 0.9}
```
Returns: `StepResult` with observation, reward, done flag, feedback.

### `GET /state`
Get current episode state snapshot.

### `POST /grade`
Deterministic episode grading (0.0–1.0).
```json
{"output": "billing refund processed within 3 business days", "task_id": "task_medium"}
```

### `GET /tasks` — List all task definitions  
### `GET /tickets` — Browse the ticket corpus  
### `GET /metrics` — Runtime episode metrics  
### `GET /health` — Server health check

Full interactive docs: `/docs`

---

## 🏗 Project Structure

```
├── app/
│   ├── env.py        # SupportEnv — reset() / step() / state()
│   ├── models.py     # Pydantic: Observation, Action, StepResult
│   ├── tasks.py      # 30 tickets, 10 workflow types, 3 task definitions
│   └── grader.py     # Deterministic scoring engine
├── server/
│   └── app.py        # FastAPI server (all endpoints)
├── ui/
│   └── index.html    # Premium glassmorphism dashboard
├── inference.py      # Full agent inference pipeline
├── openenv.yaml      # OpenEnv manifest
├── Dockerfile        # HF Spaces ready (port 7860)
└── requirements.txt
```

---

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run inference (rule-based agent, all tasks)
python inference.py

# Run with AI agent (requires HF_TOKEN)
HF_TOKEN=your_token python inference.py --mode ai --task task_hard
```

### Docker
```bash
docker build -t ai-support-env .
docker run -p 7860:7860 ai-support-env
```

---

## 📊 Multi-Dimensional Reward Design

The reward utilizes a multi-dimensional strategy, allocating points dynamically based on sub-task success with severe penalties for hallucinations and negative sentiment escalation:
- Reward = 0.3(classification) + 0.3(response quality) + 0.2(sentiment handling) + 0.2(efficiency)

| Action | Condition | Reward |
|--------|-----------|--------|
| classify | Correct category | +0.70 |
| classify | Synonym match | +0.50 |
| classify | Wrong category | −0.20 |
| respond | Empathy + resolution | +0.50 |
| respond | Partial quality | +0.15–0.35 |
| escalate | Correct escalation | +0.30 |
| escalate | False escalation | −0.20 |
| escalate | Missed escalation | −0.30 |
| close | Clean close | +0.10 |

---

## 📟 Inference Log Format

```
[START] task=task_hard env=ai-customer-support-v1 model=rule-based-v1
[STEP] step=1 action=classify reward=+0.700 cumulative=0.700 done=False
       content="complaint"
       feedback="✅ Perfect classification: 'complaint'"
[STEP] step=2 action=respond reward=+0.425 cumulative=1.125 done=False
       content="We are deeply sorry..."
[STEP] step=3 action=escalate reward=+0.300 cumulative=1.425 done=True
[END] success=True steps=3 score=0.812 rewards=[+0.700, +0.425, +0.300]
```

---

## 🏆 Scoring Criteria

- **Real-world use case** — 10 CX workflow types mapped to industry problems
- **Clean reward function** — Deterministic, keyword + intent-based scoring  
- **Proper grading** — Per-step + full-episode grading endpoints
- **Docker + HF deployment** — Port 7860, non-root user, optimized layers
- **Premium UI** — Glassmorphism dashboard with live simulation

---

## 🔮 Future Scope (Round 2)

- [x] DistilBERT Integration for offline sentiment & classification
- [x] Multi-agent Critic loops (Self-Reflecting agents)
- [x] Dynamic Curriculum Learning (Difficulty Scaling based on rolling performance limits)
- [ ] Multi-language ticket support  
- [ ] Dataset integration (Amazon reviews, support logs)

---

## 📄 License

MIT License — built for the Meta × HuggingFace OpenEnv Hackathon 2026.
