## Roadmap

### Phase 1 — AI Fundamentals (Months 1-3)
- [x] Setup secure Python environment (WSL2 / uv / Python 3.11)
- [x] First Claude API integration with proper secrets management
- [x] Token economics and model comparison (Haiku vs Sonnet)
- [x] Temperature and determinism in LLM outputs
- [x] First production-grade DevSecOps tool (CVE Triage Assistant)
- [ ] Build a local RAG with Ollama
- [ ] Deploy an LLM stack on Kubernetes

## Repository structure

ai-security-journey/
├── hello-llm/                 # First Anthropic API experiments
│   ├── main.py                # Basic Claude API call
│   └── experiments.py         # Token economics, model comparison, temperature
│
└── cve-triage-assistant/      # DevSecOps tool: CVE analysis via Claude
    ├── src/cve_triage/        # NVD client, analyzer, reporter, CLI
    └── README.md              # Detailed project documentation
