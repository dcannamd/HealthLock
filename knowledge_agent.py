# knowledge_base.py
# Manages a SECOND, SEPARATE ChromaDB collection holding published expert
# documentation — clinical guidelines, reference ranges, journal articles.
#
# WHY A SEPARATE COLLECTION:
#   Your personal health records and published medical literature are
#   fundamentally different kinds of evidence. Mixing them in one collection
#   makes it impossible to tell the model — or the user — which is which.
#   Two collections means every answer can be explicit about whether a claim
#   came from YOUR lab report or from a PUBLISHED GUIDELINE.
#
#   This distinction is the whole governance point. A guideline saying
#   "target LDL below 2.0 mmol/L" is general population advice. Your report
#   saying "LDL 3.2" is your data. Conflating them produces something that
#   sounds like a diagnosis, which this tool must never do.
#
# Uses Ollama (local LLM) — nothing leaves your machine.

import sys
import json
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
MODEL_NAME = "llama3"
EMBED_MODEL = "nomic-embed-text"

# Deliberately a DIFFERENT directory from the personal archive (./chroma_db)
KNOWLEDGE_DIR = "./chroma_db_knowledge"
KNOWLEDGE_FOLDER = "knowledge_base"
KNOWLEDGE_INDEX = "logs/knowledge_index.json"

# Larger chunks than personal documents: guidelines contain long reasoning
# passages, and cutting them mid-argument loses the meaning.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

SOURCE_TYPES = [
    "Clinical guideline",
    "Lab reference ranges",
    "Journal article",
    "Patient education",
    "Other reference",
]


# ---------------------------------------------------------------------------
# The combined-answer prompt — the governance centrepiece of this module
# ---------------------------------------------------------------------------

COMBINED_PROMPT = """You are helping a patient understand their own health documents.

You have been given TWO different kinds of source material, clearly labelled.
Treat them very differently.

[YOUR RECORDS] — the patient's own health documents. These contain THEIR
actual results, dates, and findings.

[PUBLISHED REFERENCE] — general medical literature and guidelines. These
describe what tests measure and what general population targets are. They
say NOTHING about this particular patient.

RULES YOU MUST FOLLOW:

1. Always say which kind of source a statement comes from. Use phrasing like
   "your report from 2024-06-06 shows..." versus "the published guideline
   states that in general...".

2. NEVER combine the two into a judgement about this patient's health.
   Do not say "your result is bad", "you are at risk", "this is abnormal for
   you", or anything that assesses them. You may state what their document
   says, and separately state what a guideline says, and then point out that
   only their doctor can connect the two.

3. Published guidelines describe populations, not individuals. If the patient
   asks whether their number is good, explain what the guideline says the
   general target is, note that individual targets depend on factors not in
   these documents, and direct them to their provider.

4. Use only what is below. If it is not here, say you could not find it.
   Never fill gaps with your own medical knowledge.

5. Cite sources in this format: (filename, page N). For published references
   also give the year if it appears.

6. Plain language throughout. The reader is not medically trained.

SOURCES:
{context}

Patient's question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def get_knowledge_store() -> Chroma:
    """Open (or create) the persistent knowledge-base collection."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=KNOWLEDGE_DIR,
    )


# ---------------------------------------------------------------------------
# Citation index — stores human-authored citations per file
# ---------------------------------------------------------------------------

def load_knowledge_index() -> dict:
    """Read the citation index: {filename: {source_type, citation, added}}."""
    try:
        with open(KNOWLEDGE_INDEX, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_knowledge_index(index: dict):
    Path("logs").mkdir(exist_ok=True)
    with open(KNOWLEDGE_INDEX, "w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_and_chunk(file_path: str, source_type: str, citation: str) -> list:
    """Load a reference document and chunk it with citation metadata attached."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in (".txt", ".md"):
        loader = TextLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    filename = path.name
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = filename
        chunk.metadata["source_type"] = source_type
        chunk.metadata["citation"] = citation or filename
        chunk.metadata["collection"] = "knowledge"

        raw_page = chunk.metadata.get("page")
        chunk.metadata["page_label"] = (
            (raw_page + 1) if isinstance(raw_page, int) else "n/a"
        )
        chunk.metadata["content_id"] = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:16]
        chunk.metadata["chunk_index"] = i

    print(f"  Split into {len(chunks)} chunks ({source_type}).")
    return chunks


def add_knowledge_document(
    file_path: str,
    source_type: str = "Other reference",
    citation: str = "",
) -> int:
    """
    Add one reference document to the knowledge collection.
    Returns the number of chunks added. Skips duplicates by filename.
    """
    filename = Path(file_path).name
    index = load_knowledge_index()

    if filename in index:
        print(f"  Already in knowledge base, skipping: {filename}")
        return 0

    print(f"\nAdding reference: {filename}")
    chunks = load_and_chunk(file_path, source_type, citation)

    store = get_knowledge_store()
    store.add_documents(chunks)

    index[filename] = {
        "source_type": source_type,
        "citation": citation or filename,
        "chunks": len(chunks),
        "path": str(file_path),
    }
    save_knowledge_index(index)

    print(f"  Added. Knowledge base now holds {len(index)} document(s).")
    return len(chunks)


def ingest_folder(folder: str = KNOWLEDGE_FOLDER, source_type: str = "Clinical guideline"):
    """
    Add every PDF, TXT, and MD file in a folder. Used for bulk loading.
    Citation defaults to the filename — edit them afterwards for real citations.
    """
    path = Path(folder)
    if not path.exists():
        print(f"Folder not found: {folder}")
        return []

    files = sorted(
        [p for p in path.iterdir()
         if p.suffix.lower() in (".pdf", ".txt", ".md")]
    )
    if not files:
        print(f"No PDF, TXT, or MD files found in {folder}/")
        return []

    print(f"Found {len(files)} file(s) in {folder}/")
    added = []
    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}]")
        try:
            n = add_knowledge_document(str(f), source_type=source_type)
            if n:
                added.append((f.name, n))
        except Exception as e:
            print(f"  FAILED: {f.name} — {e}")
    return added


# ---------------------------------------------------------------------------
# Listing and deletion
# ---------------------------------------------------------------------------

def list_knowledge_documents() -> list:
    """Return [{filename, source_type, citation, chunks}] for the UI."""
    index = load_knowledge_index()
    return [
        {
            "filename": fname,
            "source_type": meta.get("source_type", "Other reference"),
            "citation": meta.get("citation", fname),
            "chunks": meta.get("chunks", 0),
        }
        for fname, meta in sorted(index.items())
    ]


def get_knowledge_choices() -> list:
    """Return (label, filename) tuples for a UI selector."""
    return [
        (f"{d['filename']}  —  {d['source_type']}", d["filename"])
        for d in list_knowledge_documents()
    ]


def format_knowledge_list() -> str:
    """Human-readable listing of the knowledge base."""
    docs = list_knowledge_documents()
    if not docs:
        return (
            "No reference documents yet.\n\n"
            f"Put PDFs in the `{KNOWLEDGE_FOLDER}/` folder, then use "
            "'Load everything in knowledge_base folder' below."
        )

    total = sum(d["chunks"] for d in docs)
    lines = [f"{len(docs)} reference document(s), {total} indexed passages.\n"]
    for d in docs:
        lines.append(
            f"{d['filename']}\n"
            f"    {d['source_type']}  ·  {d['chunks']} passages\n"
            f"    cited as: {d['citation']}"
        )
    return "\n\n".join(lines)


def update_citation(filename: str, citation: str) -> bool:
    """Set the human-authored citation for a file. Affects future answers."""
    index = load_knowledge_index()
    if filename not in index:
        return False
    index[filename]["citation"] = citation
    save_knowledge_index(index)
    return True


def delete_knowledge_document(filename: str) -> str:
    """Remove a reference document from the collection and the index."""
    notes = []
    try:
        store = get_knowledge_store()
        data = store.get(where={"source_file": filename})
        ids = data.get("ids", [])
        if ids:
            store.delete(ids=ids)
            notes.append(f"removed {len(ids)} passage(s)")
        else:
            notes.append("no passages found")
    except Exception as e:
        notes.append(f"removal failed: {e}")

    index = load_knowledge_index()
    if filename in index:
        del index[filename]
        save_knowledge_index(index)
        notes.append("removed from index")

    return "; ".join(notes)


# ---------------------------------------------------------------------------
# Retrieval and answering
# ---------------------------------------------------------------------------

def retrieve_knowledge(question: str, k: int = 4) -> list:
    """Retrieve relevant passages from the knowledge base."""
    try:
        store = get_knowledge_store()
        retriever = store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(question)
    except Exception as e:
        print(f"Knowledge retrieval failed: {e}")
        return []


def retrieve_personal(personal_store, question: str, selected_files: list, k: int = 5) -> list:
    """Retrieve relevant chunks from the patient's own documents."""
    if not personal_store:
        return []

    search_kwargs = {"k": k}
    if selected_files:
        if len(selected_files) == 1:
            search_kwargs["filter"] = {"source_file": selected_files[0]}
        else:
            search_kwargs["filter"] = {"source_file": {"$in": selected_files}}

    try:
        retriever = personal_store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(question)
    except Exception as e:
        print(f"Personal retrieval failed: {e}")
        return []


def ask_with_knowledge(
    personal_store,
    question: str,
    selected_files: list = None,
    include_knowledge: bool = True,
    k_personal: int = 5,
    k_knowledge: int = 4,
) -> dict:
    """
    Answer a question using the patient's documents, optionally supplemented
    by published references.

    Returns {answer, personal_sources, knowledge_sources, used_knowledge}.
    """
    personal_chunks = retrieve_personal(
        personal_store, question, selected_files or [], k=k_personal
    )
    knowledge_chunks = retrieve_knowledge(question, k=k_knowledge) if include_knowledge else []

    if not personal_chunks and not knowledge_chunks:
        return {
            "answer": "I couldn't find anything relevant in your documents.",
            "personal_sources": [],
            "knowledge_sources": [],
            "used_knowledge": False,
        }

    parts = []

    for chunk in personal_chunks:
        m = chunk.metadata
        parts.append(
            f"[YOUR RECORDS | {m.get('source_file', 'unknown')} "
            f"| dated {m.get('date', 'unknown')} "
            f"| page {m.get('page_label', 'n/a')}]\n{chunk.page_content}"
        )

    for chunk in knowledge_chunks:
        m = chunk.metadata
        parts.append(
            f"[PUBLISHED REFERENCE | {m.get('citation', m.get('source_file', 'unknown'))} "
            f"| {m.get('source_type', 'reference')} "
            f"| page {m.get('page_label', 'n/a')}]\n{chunk.page_content}"
        )

    context = "\n\n---\n\n".join(parts)

    print(f"\nRetrieved {len(personal_chunks)} from your records, "
          f"{len(knowledge_chunks)} from references.")
    for c in personal_chunks:
        print(f"  YOURS: {c.metadata.get('source_file')} "
              f"p.{c.metadata.get('page_label', 'n/a')}")
    for c in knowledge_chunks:
        print(f"  REF:   {c.metadata.get('source_file')} "
              f"p.{c.metadata.get('page_label', 'n/a')}")

    llm = OllamaLLM(model=MODEL_NAME)
    answer = llm.invoke(
        COMBINED_PROMPT.format(context=context, question=question)
    ).strip()

    def describe(chunks, kind):
        out = []
        seen = set()
        for c in chunks:
            m = c.metadata
            key = (m.get("source_file"), m.get("page_label"))
            if key in seen:
                continue
            seen.add(key)
            label = (
                m.get("citation") if kind == "ref"
                else m.get("source_file", "unknown")
            )
            page = m.get("page_label", "n/a")
            out.append(f"{label} (page {page})")
        return out

    return {
        "answer": answer,
        "personal_sources": describe(personal_chunks, "personal"),
        "knowledge_sources": describe(knowledge_chunks, "ref"),
        "used_knowledge": bool(knowledge_chunks),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python knowledge_base.py ingest [folder] [source_type]")
        print("  python knowledge_base.py list")
        print("  python knowledge_base.py delete <filename>")
        print("  python knowledge_base.py ask \"<question>\"")
        print()
        print("Source types:", ", ".join(SOURCE_TYPES))
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        folder = sys.argv[2] if len(sys.argv) > 2 else KNOWLEDGE_FOLDER
        stype = sys.argv[3] if len(sys.argv) > 3 else "Clinical guideline"
        added = ingest_folder(folder, stype)
        print(f"\nAdded {len(added)} new document(s).")
        if added:
            print("\nNote: citations default to filenames. Set proper citations "
                  "in the app's Knowledge base tab so answers are traceable.")

    elif command == "list":
        print()
        print(format_knowledge_list())
        print()

    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python knowledge_base.py delete <filename>")
            sys.exit(1)
        print(delete_knowledge_document(sys.argv[2]))

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python knowledge_base.py ask \"<question>\"")
            sys.exit(1)
        chunks = retrieve_knowledge(sys.argv[2], k=4)
        if not chunks:
            print("Nothing relevant found in the knowledge base.")
        else:
            print(f"\n{len(chunks)} relevant passage(s):\n")
            for c in chunks:
                m = c.metadata
                print(f"--- {m.get('citation')} (page {m.get('page_label')}) ---")
                print(c.page_content[:400].strip())
                print()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)