# 🛡️ Local RAG for DevSecOps

> Privacy-first Retrieval Augmented Generation (RAG) over local DevSecOps documentation. 100% local, zero data sent to external APIs.

## What it does

Ask questions in natural language about your private DevSecOps documents (PDFs, Markdown, plaintext) and get answers grounded in those documents — with source citations.

The whole pipeline runs **locally on your machine**:
- 🧠 **LLM**: [Ollama](https://ollama.com) (Qwen, Llama, Mistral, etc.)
- 🔍 **Embeddings**: [sentence-transformers](https://www.sbert.net/) (multilingual)
- 🗄️ **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- 🖥️ **CLI**: [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/)

No data ever leaves your machine. Suitable for sensitive runbooks, internal advisories, or any document you can't send to a cloud LLM.

## Why this project

In 2026, organizations need RAG systems for their internal documentation but face real constraints:
- **Regulatory** (GDPR, banking secrecy, defense): cannot send data to OpenAI/Anthropic
- **Cost**: cloud LLMs scale linearly with usage
- **Privacy**: prompts and outputs may contain trade secrets

This project demonstrates a **fully local RAG architecture** that addresses all three. It also serves as a foundation for **adversarial testing** — the same RAG you build can be attacked to learn AI security in practice.

## Architecture

┌───────────────────────────────────────────┐
             │              YOUR MACHINE                  │
             │                                            │
             
docs/         │   ┌─────────────┐    ┌──────────────┐    │
├── pdf  ─────┼──►│   ingest    │───►│  ChromaDB    │    │
├── md        │   │  (chunker + │    │  (vectors)   │    │
└── txt       │   │  embedder)  │    └──────┬───────┘    │
│   └─────────────┘           │            │
│                              │            │
user question │   ┌─────────────┐           │            │
─────────────┼──►│  retriever  │◄──────────┘            │
│   │  (semantic  │                        │
│   │   search)   │                        │
│   └──────┬──────┘                        │
│          │                                │
│          ▼                                │
│   ┌─────────────┐    ┌──────────────┐    │
│   │  pipeline   │───►│   Ollama     │    │
│   │  (augment   │    │  (local LLM) │    │
│   │   prompt)   │◄───│              │    │
│   └──────┬──────┘    └──────────────┘    │
│          │                                │
│          ▼                                │
│     Answer +                              │
│     Sources                               │
└───────────────────────────────────────────┘

## Installation

### Prerequisites

- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **Ollama** ([install](https://ollama.com))
- At least **8 GB RAM** (16 GB recommended)

### Setup

```bash
git clone https://github.com/SofieneLasmarHazgui/ai-security-journey.git
cd ai-security-journey/local-rag-devsecops
uv sync
```

Pull a small LLM model via Ollama:

```bash
ollama pull qwen2.5:3b   # ~2 GB, runs well on CPU
```

Optional — copy the env template:

```bash
cp .env.example .env
```

## Usage

### 1. Add your documents

Drop PDFs, Markdown, or TXT files into the `docs/` folder.
docs/
├── falco_runbook.md
├── owasp_llm_top10.pdf
└── kubernetes_security_guide.md

### 2. Index them

```bash
uv run rag ingest
```

The tool chunks each document, generates embeddings locally, and stores everything in ChromaDB at `./vectordb/`.

### 3. Ask questions

```bash
uv run rag ask "How does Falco detect shells in containers?"
```

You'll see:
- Sources identified (with similarity scores)
- The answer streaming token by token
- Performance stats (time to first token, total time)

### Other commands

```bash
uv run rag info                              # Stats about your indexed corpus
uv run rag ask "..." --no-stream             # Wait for full response
uv run rag ask "..." --top-k 3               # Use top 3 chunks
uv run rag ask "..." --threshold 0.4         # Stricter relevance filter
uv run rag ask "..." --model llama3.2:1b     # Use a different Ollama model
uv run rag --help                            # Full help
```

## Project structure
local-rag-devsecops/
├── src/rag/
│   ├── ingest.py       # Document chunking + embedding + indexing
│   ├── retriever.py    # Semantic search over ChromaDB
│   ├── llm.py          # Ollama API client (generate + stream)
│   ├── pipeline.py     # Augmented prompt orchestration
│   └── cli.py          # Typer-based command-line interface
├── docs/               # Your documents (gitignored except README)
├── vectordb/           # ChromaDB storage (gitignored)
└── pyproject.toml
## Security design choices

This RAG is designed with AI Security best practices in mind:

- **Strict similarity threshold** (default 0.3) — refuses to answer when no relevant context is found, preventing hallucinations on out-of-scope questions
- **Source citation by design** — every chunk is tagged with source filename, chunk index, and content hash (auditability for compliance like EU AI Act)
- **Content hash per chunk** (SHA-256) — detects unauthorized modifications to indexed documents
- **Low default temperature** (0.2) — favors reproducible, auditable responses
- **Explicit system prompt** — instructs the LLM to refuse out-of-context answers and to never combine sources improperly
- **No external API calls** — by design, no data is sent to OpenAI, Anthropic, or any third-party LLM provider
- **Defensive output handling** — exceptions raised for empty queries, missing collections, unavailable models, network timeouts

These patterns map directly to:
- **OWASP LLM Top 10** — particularly LLM01 (Prompt Injection), LLM06 (Sensitive Information Disclosure), LLM08 (Excessive Agency)
- **MITRE ATLAS** — data poisoning and inference attack mitigations
- **NIST AI RMF** — auditability and transparency requirements

## Performance notes

Inference runs on CPU by default. Expected performance on a typical laptop (16 GB RAM, no GPU):

| Model       | Size  | Tokens/sec | First token | 200-token response |
|-------------|-------|-----------:|------------:|-------------------:|
| qwen2.5:3b  | 1.9 GB |  4-8       |   1-3 s     |  30-50 s            |
| gemma2:2b   | 1.6 GB |  5-10      |   1-2 s     |  20-40 s            |
| llama3.2:1b | 0.8 GB |  10-20     |   <1 s      |  10-20 s            |

For better performance, use a machine with GPU (NVIDIA or Apple Silicon) — Ollama auto-detects and uses it.

## Roadmap

This project is part of a hands-on learning journey toward AI Security consulting. Planned extensions:

- [ ] Adversarial testing notebook (prompt injection scenarios against this RAG)
- [ ] Output guardrails layer (filter responses with regex / classifier)
- [ ] Input sanitization (detect injection attempts in user queries)
- [ ] Kubernetes deployment manifests (production-ready hardening)
- [ ] Audit logging (every query + retrieved sources + response, for compliance)

## License

MIT — see root repository.

## Author

Built by Sofiene Lasmar as part of an [AI Security learning journey](../README.md).

DevSecOps engineer transitioning to AI Security / MLSecOps consulting.
