# tagger_agent.py
# Reads a PDF or TXT file and classifies it into one of 8 health document categories.
# Also extracts the document date for chronological tracking.
# Uses Ollama (local LLM) — nothing leaves your machine.

import sys
import re
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

CATEGORY_PROMPT = """You are a medical document classifier. 
Classify the following document into exactly one of these categories:
{categories}

Respond with ONLY the category name. No explanation. No punctuation.

Document:
{text}

Category:"""

DATE_PROMPT = """Find the date this document is FROM (the date of service, test, or visit — not today's date).
Respond with ONLY the date in YYYY-MM-DD format. If multiple dates appear, use the most prominent one (e.g. test date, visit date).
If no date is found, respond with exactly: UNKNOWN

Document:
{text}

Date:"""


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


def extract_date(text: str, llm: OllamaLLM) -> str:
    """Ask the model to find the document's date. Returns YYYY-MM-DD or 'UNKNOWN'."""
    prompt = DATE_PROMPT.format(text=text[:1500])
    response = llm.invoke(prompt).strip()

    # Validate format with regex — reject anything that's not YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", response):
        return response
    return "UNKNOWN"


def tag_document(file_path: str) -> dict:
    """Load a document and return its predicted category and extracted date."""
    print(f"\nLoading document: {file_path}")
    text = load_document(file_path)
    text_preview = text[:2000]

    print(f"Document loaded ({len(text)} chars). Sending to {MODEL_NAME}...")

    llm = OllamaLLM(model=MODEL_NAME)

    # Classify category
    category_prompt = CATEGORY_PROMPT.format(
        categories="\n".join(f"- {c}" for c in CATEGORIES),
        text=text_preview,
    )
    category = llm.invoke(category_prompt).strip()
    if category not in CATEGORIES:
        print(f"Warning: model returned '{category}' — falling back to 'Other'")
        category = "Other"

    # Extract date
    doc_date = extract_date(text, llm)

    result = {
        "file": file_path,
        "category": category,
        "date": doc_date,
        "chars_processed": len(text_preview),
    }

    print(f"\n--- RESULT ---")
    print(f"File     : {result['file']}")
    print(f"Category : {result['category']}")
    print(f"Date     : {result['date']}")

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tagger_agent.py <path_to_document>")
        sys.exit(1)

    tag_document(sys.argv[1])
