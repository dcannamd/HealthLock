# chat_agent.py
# Allows a patient to ask questions across MULTIPLE health documents.
# Uses ChromaDB for PERSISTENT local vector storage and RAG (Retrieval Augmented Generation)
# to ground answers in the actual document content.
# Nothing leaves your machine.

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
MODEL_NAME = "llama3"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"

PROMPT_TEMPLATE = """You are a helpful medical document assistant.
Answer the patient's question using ONLY the information in the documents provided below.
Each document chunk includes its source file and date so you can reference WHEN something happened.
Use plain, simple language a non-medical person can understand.
If asked about trends over time (improving, worsening, changes), compare values across the different dates shown.
If the answer is not in the documents, say "I couldn't find that information in your documents."
Never guess or make up medical information.

Documents (with source and date):
{context}

Patient's question: {question}

Answer:"""


def load_and_chunk_document(file_path: str, category: str = "Unknown", doc_date: str = "UNKNOWN") -> list:
    """Load a document, split it into chunks, and attach metadata to each chunk."""
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)

    # Attach metadata to every chunk so we can filter/reference later
    filename = path.name
    for chunk in chunks:
        chunk.metadata["source_file"] = filename
        chunk.metadata["category"] = category
        chunk.metadata["date"] = doc_date

    print(f"Document split into {len(chunks)} chunks (date: {doc_date}, category: {category}).")
    return chunks


def get_vector_store() -> Chroma:
    """Open the persistent vector store (creates it if it doesn't exist yet)."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def add_document_to_store(file_path: str, category: str = "Unknown", doc_date: str = "UNKNOWN") -> Chroma:
    """Add a new document's chunks to the PERSISTENT vector store. Does not erase existing documents."""
    print(f"\nAdding to archive: {file_path}")
    chunks = load_and_chunk_document(file_path, category, doc_date)

    vectorstore = get_vector_store()
    vectorstore.add_documents(chunks)
    print(f"Added. Archive now includes this document permanently.")
    return vectorstore


def ask_question(vectorstore: Chroma, question: str, k: int = 5, selected_files: list = None) -> str:
    """Retrieve relevant chunks and generate a grounded answer.
    If selected_files is provided, only search within those documents."""
    search_kwargs = {"k": k}
    if selected_files:
        if len(selected_files) == 1:
            search_kwargs["filter"] = {"source_file": selected_files[0]}
        else:
            search_kwargs["filter"] = {"source_file": {"$in": selected_files}}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        return "I couldn't find any relevant information in your documents."

    # Build context, labeling each chunk with its source and date
    context_parts = []
    for chunk in relevant_chunks:
        source = chunk.metadata.get("source_file", "unknown file")
        date = chunk.metadata.get("date", "unknown date")
        context_parts.append(f"[Source: {source} | Date: {date}]\n{chunk.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    print(f"\nRetrieved {len(relevant_chunks)} relevant chunks.")
    for chunk in relevant_chunks:
        print(f"  - {chunk.metadata.get('source_file')} ({chunk.metadata.get('date')})")

    llm = OllamaLLM(model=MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt).strip()

    return answer


def get_document_choices(vectorstore: Chroma) -> list:
    """Return a list of (display_label, filename) tuples for use in a UI selector."""
    docs = list_archive_documents(vectorstore)
    docs.sort(key=lambda d: d.get("date", ""), reverse=True)
    choices = []
    for doc in docs:
        label = f"{doc['filename']} — {doc['category']} ({doc['date']})"
        choices.append((label, doc["filename"]))
    return choices


def list_archive_documents(vectorstore: Chroma) -> list:
    """Return a list of unique documents currently in the archive."""
    try:
        data = vectorstore.get()
        seen = {}
        for metadata in data.get("metadatas", []):
            fname = metadata.get("source_file", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "category": metadata.get("category", "Unknown"),
                    "date": metadata.get("date", "UNKNOWN"),
                }
        return list(seen.values())
    except Exception as e:
        print(f"Could not list archive: {e}")
        return []


def chat_session(file_path: str):
    """Run an interactive chat session — adds the document to the archive, then chats across ALL archived documents."""
    vectorstore = add_document_to_store(file_path)

    docs = list_archive_documents(vectorstore)
    print(f"\nYour archive currently contains {len(docs)} document(s):")
    for doc in docs:
        print(f"  - {doc['filename']} | {doc['category']} | {doc['date']}")

    print("\n--- CHAT SESSION ---")
    print("Ask questions across ALL your documents. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Ending chat session.")
            break

        if not question:
            continue

        print("\nThinking...\n")
        answer = ask_question(vectorstore, question)
        print(f"Answer: {answer}\n")
        print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chat_agent.py <path_to_document>")
        sys.exit(1)

    chat_session(sys.argv[1])