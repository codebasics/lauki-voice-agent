# Lauki Phones Voice Agent 🎙️📱

## Voice Agent

A real-time voice-based customer support agent for **Lauki Phones** — a fictional Indian telecom company. Customers speak to an AI assistant over a live audio stream that can look up account balances, answer plan questions using RAG over PDF docs, check network coverage, recommend plans, and escalate to human support. Built with **LiveKit Agents**, **LangChain + Groq**, **Docling**, and **Qdrant**.

---

## Getting Started

### Prerequisites

- Python `>= 3.12`
- [LiveKit Cloud](https://livekit.io/) credentials (URL, API Key, API Secret)
- [Groq](https://console.groq.com/keys) API Key

### Installation

```bash
# Install dependencies
uv sync  # or: pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your LiveKit and Groq credentials

# Activate virtual env
source .venv/bin/activate

# Ingest PDF documents into vector DB
python ingest.py

# Start the voice agent
python -m voice_agent.main console
```


---

## Example Prompts

| Try saying… | What happens |
|---|---|
| *"What plans do you offer?"* | Searches plan docs via RAG |
| *"What's the 5G coverage in Bangalore?"* | Returns network status by region |
| *"I want to speak to a human"* | Creates a support ticket and escalates |

---
Copyright©️ Codebasics Inc. All rights reserved.
