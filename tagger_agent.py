# tagger_agent.py
# Reads a PDF or TXT file and classifies it into one of 8 health document categories.
# Uses Ollama (local LLM) — nothing leaves your machine.

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM

# --- Configuration ---
MODEL_NAME = "llama3"

CATEGORIES = [
    "Lab Results",
    "Prescription",
    "Appointment Summary",
    "Insurance / EOB",
    "Medical History",
    "Imaging Report",
    "Referral",
    "Other",
]

PROMPT_TEMPLATE = """You are a medical document classifier. 
Classify the following document into exactly one of these categories:
{categories}

Respond with ONLY the category name. No explanation. No punctuation.

Document:
{text}

Category:"""


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


def tag_document(file_path: str) -> dict:
    """Load a document and return its predicted category."""
    print(f"\nLoading document: {file_path}")
    text = load_document(file_path)

    # Truncate to first 2000 chars — enough context, avoids token limits
    text_preview = text[:2000]

    print(f"Document loaded ({len(text)} chars). Sending to {MODEL_NAME}...")

    llm = OllamaLLM(model=MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(
        categories="\n".join(f"- {c}" for c in CATEGORIES),
        text=text_preview,
    )

    category = llm.invoke(prompt).strip()

    # Validate — if response isn't a known category, fall back to Other
    if category not in CATEGORIES:
        print(f"Warning: model returned '{category}' — falling back to 'Other'")
        category = "Other"

    result = {
        "file": file_path,
        "category": category,
        "chars_processed": len(text_preview),
    }

    print(f"\n--- RESULT ---")
    print(f"File     : {result['file']}")
    print(f"Category : {result['category']}")
    print(f"Chars    : {result['chars_processed']}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tagger_agent.py <path_to_document>")
        sys.exit(1)

    tag_document(sys.argv[1])