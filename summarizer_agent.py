# summarizer_agent.py
# Reads a health document and returns:
# 1. A plain-language summary
# 2. Key values (lab numbers, medications, dates)
# 3. Action items for the patient
# Uses Ollama (local LLM) — nothing leaves your machine.

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM

# --- Configuration ---
MODEL_NAME = "llama3"

PROMPT_TEMPLATE = """You are a helpful medical document assistant.
A patient has uploaded a health document and needs help understanding it.
Respond in plain, simple language a non-medical person can understand.

Analyze the document below and respond in exactly this format:

SUMMARY:
(2-3 sentences explaining what this document is and what it means for the patient)

KEY VALUES:
(List the most important numbers, medications, dates, or findings. One per line, starting with a dash.)

ACTION ITEMS:
(List what the patient should do next. One per line, starting with a dash. If none, write "None identified.")

---
Document:
{text}
"""


def load_document(file_path: str) -> str:
    """Load a PDF or TXT file and return its text content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    pages = loader.load()
    return "\n".join(page.page_content for page in pages)


def summarize_document(file_path: str) -> dict:
    """Load a document and return a structured summary."""
    print(f"\nLoading document: {file_path}")
    text = load_document(file_path)

    # Truncate to first 3000 chars — more context than tagger needs
    text_preview = text[:3000]

    print(f"Document loaded ({len(text)} chars). Sending to {MODEL_NAME}...")
    print("Generating summary — this may take 20-40 seconds on first run...\n")

    llm = OllamaLLM(model=MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(text=text_preview)
    response = llm.invoke(prompt).strip()

    # Parse the three sections
    sections = {"summary": "", "key_values": "", "action_items": ""}

    current_section = None
    lines = response.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("SUMMARY:"):
            current_section = "summary"
            remainder = line.replace("SUMMARY:", "").strip()
            if remainder:
                sections["summary"] += remainder + " "
        elif line.startswith("KEY VALUES:"):
            current_section = "key_values"
        elif line.startswith("ACTION ITEMS:"):
            current_section = "action_items"
        elif line == "---":
            current_section = None
        elif current_section and line:
            sections[current_section] += line + "\n"

    # Clean up
    for key in sections:
        sections[key] = sections[key].strip()

    result = {
        "file": file_path,
        "summary": sections["summary"],
        "key_values": sections["key_values"],
        "action_items": sections["action_items"],
        "chars_processed": len(text_preview),
    }

    print("--- RESULT ---")
    print(f"File     : {result['file']}")
    print(f"\nSUMMARY:\n{result['summary']}")
    print(f"\nKEY VALUES:\n{result['key_values']}")
    print(f"\nACTION ITEMS:\n{result['action_items']}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarizer_agent.py <path_to_document>")
        sys.exit(1)

    summarize_document(sys.argv[1])