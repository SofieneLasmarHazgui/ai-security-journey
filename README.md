# ai-security-journey
My learning journey from DevSecOps to AI Security / MLSecOps. Practical experiments, security analyses, and project notes.

# AI Security Journey 🛡️🤖

> Personal learning journey from DevSecOps to AI Security / MLSecOps engineering.

## About this repository

I'm **Sofiene Lasmar**, a DevSecOps engineer with 10+ years of experience, currently transitioning into **AI Security** and **MLSecOps**. This repository tracks my hands-on learning journey from fundamentals to production-ready AI security skills.

The goal: build deep technical expertise in securing AI systems — from LLM applications and RAG pipelines to MLOps infrastructure — combining traditional security engineering with the new challenges introduced by generative AI.

## Why this matters

Every week, more organizations deploy LLM-powered applications without proper security controls. The OWASP Top 10 for LLM Applications and MITRE ATLAS describe a rapidly evolving threat landscape, while regulations like the **EU AI Act** and **NIST AI RMF** are creating real compliance pressure.

This repo is my public lab to:
- Master AI security concepts through practical projects
- Document attacks, defenses, and mitigations with reproducible code
- Build a portfolio of audits, red-team exercises, and hardening playbooks

## Roadmap

### Phase 1 — AI Fundamentals (Months 1-3)
- [x] Setup secure Python environment (WSL2 / uv / Python 3.11)
- [x] First Claude API integration with proper secrets management
- [x] Token economics and model comparison (Haiku vs Sonnet)
- [x] Temperature and determinism in LLM outputs
- [ ] Build a local RAG with Ollama
- [ ] Deploy an LLM stack on Kubernetes

### Phase 2 — Applied AI Security (Months 4-6)
- [ ] OWASP LLM Top 10 — hands-on exploitation lab
- [ ] MITRE ATLAS — mapping attacks to techniques
- [ ] Automated red-teaming pipelines (Garak, PyRIT)
- [ ] Hugging Face supply chain security audit

### Phase 3 — Governance & Compliance (Months 7-9)
- [ ] EU AI Act practical implementation guide
- [ ] NIST AI RMF risk mapping templates
- [ ] ISO/IEC 42001 implementation notes

### Phase 4 — Freelance & Specialization (Months 10-12)
- [ ] Public case study: full audit of an open-source AI project
- [ ] Open-source AI security tooling

## Repository structure
ai-security-journey/
├── hello-llm/             # First API experiments (Anthropic Claude)
│   ├── main.py            # Basic Claude API call
│   └── experiments.py     # Token economics, model comparison
└── (more projects coming)
