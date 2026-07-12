# 🏥 HealthLock
### Local AI Health Document Organizer

> A fully local, privacy-first AI pipeline that lets patients upload, organize, summarize, and chat with their personal health documents. **Nothing leaves your computer.**

---

## What Is This?

HealthLock is a personal health document assistant powered by a local large language model (Llama 3). It runs entirely on your own machine — no cloud, no subscriptions, no data sharing. Upload your lab results, prescriptions, and appointment summaries, and ask questions about them in plain English.

This project also serves as a teaching artifact demonstrating how local AI agent pipelines work, with AI governance and Human-in-the-Loop (HITL) principles built in from the ground up.

---

## Features

- **Automatic tagging** — classifies documents into 8 health categories (Lab Results, Prescription, Appointment Summary, Insurance/EOB, Medical History, Imaging Report, Referral, Other)
- **Date extraction** — identifies the document date automatically for chronological tracking
- **Plain-language summaries** — explains what a document means in terms anyone can understand
- **Trend analysis** — select multiple documents to compare values over time (e.g. cholesterol across 7 years of lab results)
- **RAG chat** — ask questions grounded in your actual documents, not model hallucinations
- **Multi-document chat** — select which documents to include in each chat session
- **Batch upload** — drag and drop multiple PDFs at once
- **Persistent archive** — documents are stored in a local vector database that survives app restarts
- **Human correction** — override the AI's category assignment with one click
- **Flag for review** — mark any AI output for human follow-up
- **Local audit log** — every AI action is logged to a JSON file, exportable as CSV
- **AI disclosure** — governance guardrails on every output

---

## Privacy Guarantee

All processing happens locally on your computer using Ollama. No document content, query, or result is ever sent to the internet. The vector database (`chroma_db/`) and audit log (`logs/`) are stored only on your machine.

---

## Tech Stack

| Component | Purpose |
|---|---|
| [Ollama](https://ollama.com) | Local LLM runtime |
| [Llama 3](https://ollama.com/library/llama3) | Main language model (tagging, summarization, chat) |
| [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | Embedding model for vector search |
| [LangChain](https://python.langchain.com) | Document loading, chunking, RAG |
| [ChromaDB](https://www.trychroma.com) | Local persistent vector store |
| [Gradio](https://gradio.app) | Browser-based UI |
| Python 3.9+ | Runtime |

---

## AI Agents

### `tagger_agent.py`
Classifies a document into one of 8 health categories and extracts the document date. Reads the first ~2000 characters and sends a structured prompt to Llama 3.

### `summarizer_agent.py`
Two modes:
- **Single document** — returns a plain-language summary, key values, and action items
- **Multi-document trend** — compares values across multiple documents over time, highlighting changes and things to discuss with your doctor

### `chat_agent.py`
RAG-based question answering. Searches ChromaDB for the most relevant chunks from your selected documents, then grounds the model's answer in that retrieved content. Supports filtering by specific documents so you can scope your questions to a subset of your archive.

---

## Governance & Human-in-the-Loop Features

HealthLock is built around responsible AI principles aligned with the NIST AI Risk Management Framework (AI RMF) and EU AI Act requirements for transparency and human oversight:

| Feature | Framework Alignment |
|---|---|
| AI Disclosure banner on every screen | EU AI Act Art. 13 — Transparency |
| Confidence indicator (Low/Medium/High) | NIST AI RMF — Measure |
| Human category correction | EU AI Act Art. 14 — Human Oversight |
| Flag for review button | EU AI Act Art. 14 — Human Oversight |
| Local audit log (JSON + CSV export) | NIST AI RMF — Govern / EU AI Act Art. 12 |
| Source transparency in chat (chunk attribution) | NIST AI RMF — Measure |
| Plain-language guardrails on every output | NIST AI RMF — Govern |

---

## Requirements

- macOS 11 (Big Sur) or later — tested on macOS Sequoia 15 with 16GB RAM
- Python 3.9 or later
- [Ollama](https://ollama.com/download) installed and running

---

## Setup

**1. Install Ollama and pull the required models:**
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

**2. Clone the repository:**
```bash
git clone https://github.com/dcannamd/HealthLock.git
cd HealthLock
```

**3. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

**5. Launch the app:**
```bash
python app.py
```

**6. Open your browser:**
```
http://127.0.0.1:7860
```

---

## How to Use

**Step 1 — Upload & Tag**
Drag one or more PDF or TXT health documents into the upload box and click **Upload & Tag**. Each document is automatically classified and dated, then added to your permanent local archive. The Document History panel shows everything processed this session.

**Step 2 — Summarize**
Switch to the Summarize tab. Click **Refresh Document List** to see your archive, then:
- Select **one document** for an individual plain-language summary
- Select **multiple documents** for a trend analysis comparing values over time

**Step 3 — Chat**
Switch to the Chat tab. Select which documents to include, then ask questions in plain English. Answers are grounded only in your selected documents — the model will not guess or make things up. Example questions:
- *"What was my cholesterol level in 2023?"*
- *"Is my blood pressure improving over time?"*
- *"When is my next follow-up appointment?"*

**Step 4 — Audit Log**
Every AI action is recorded in the Audit Log tab. Click **Refresh Log** to view it, or **Export as CSV** to save a copy.

---

## Project Structure

```
HealthLock/
├── app.py                  # Gradio UI — ties all agents together
├── tagger_agent.py         # Classifies documents + extracts dates
├── summarizer_agent.py     # Single-doc summary + multi-doc trend analysis
├── chat_agent.py           # RAG chat with persistent vector store
├── audit_log.py            # Local governance audit logging
├── requirements.txt        # Python dependencies
├── chroma_db/              # Persistent local vector database (gitignored)
├── documents/              # Your health documents (gitignored)
├── logs/                   # Audit log files (gitignored)
└── sample_documents/       # Fictional test documents for pipeline testing
```

---

## Known Limitations

- Summarize trend analysis requires documents to be in the current session's upload history (re-upload if you restart the app)
- Intel Mac users will experience slower inference (~20-60 seconds per operation) compared to Apple Silicon
- The model occasionally misreads lab values as abnormal when they are within normal range — always verify with your healthcare provider
- Gradio 3.x is required for Python 3.9 compatibility

---

## Sample Test Documents

The `sample_documents/` folder contains fictional health documents for testing the pipeline. Do not place real health documents there. Real documents go in the `documents/` folder, which is gitignored.

---

## Disclaimer

HealthLock is a personal productivity tool, not a medical device. AI-generated summaries and answers can contain errors. **Do not use this as a substitute for professional medical advice.** Always verify important information with your healthcare provider.

---

## Research Context

This project is part of a broader research initiative on AI/ML applications for patient empowerment and health literacy. It is designed to be teachable to non-technical users using Ollama as the local LLM platform, and to demonstrate real-world AI governance principles in an accessible, patient-facing context.

---

## License

MIT License — free to use, modify, and share.

---

*Built with Ollama + Llama 3 + LangChain + ChromaDB + Gradio. Runs entirely on your computer.*
