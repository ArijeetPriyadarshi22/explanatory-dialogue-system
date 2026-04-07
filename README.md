# MCP Client-Server Dialogue System

A  Python project that implements an orchestrator and multiple servers for a client-server dialogue/knowledge system. It includes domain knowledge JSON files, an orchestrator entrypoint, and separate servers for knowledge, logging, and explainability (XAI).

## Repository structure

- `requirements.txt` — Python dependencies used by the project.
- `orchestrator/main.py` — Orchestrator entrypoint that drives dialogue sessions.
- `orchestrator/logs/` — Saved dialogue session logs.
- `orchestrator/logs/session_<timestamp>/dialogue_log.txt` — Example dialogue logs recorded per session.
- `servers/knowledge_server.py` — Knowledge server that loads domain knowledge and answers queries.
- `servers/logger_server.py` — Logger server that accepts and persists dialogue logs.
- `servers/xai_server.py` — (XAI) Explainability server, provides traceability or explanations for decisions.
- `domain_knowledge/` — JSON domain knowledge files used by the knowledge server:
  - `credit_knowledge.json`
  - `diabetes.json`
  - `titanic_knowledge.json`


## Goals / Overview

This project demonstrates a modular client-server dialogue setup where:
- Domain knowledge is stored in JSON and served by the knowledge server.
- An orchestrator coordinates dialogue sessions and interacts with servers.
- Dialogue sessions and decisions are logged for auditing and debugging.
- An XAI server exposes explainability traces or rationale for decisions.


## Requirements

- Python 3.8+ recommended
- pip

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Quick start — run the orchestrator

From the repository root, run the orchestrator (this will use the servers implemented in `servers/`):

```bash
# activate your venv first (see above)
python orchestrator/main.py
```

## How the pieces fit together (high-level)

- Orchestrator (`orchestrator/main.py`): drives the dialogue flow. It loads domain knowledge (or calls the knowledge server), routes user turns to the appropriate server, and records to the logger server.
- Knowledge server (`servers/knowledge_server.py`): exposes domain knowledge queries (e.g., lookup facts, policies, or response templates) backed by JSON files in `domain_knowledge/`.
- Logger server (`servers/logger_server.py`): receives logs (dialogue turns, decisions) and writes them to `orchestrator/logs/`.
- XAI server (`servers/xai_server.py`): produces explanations for decisions/actions when requested.

