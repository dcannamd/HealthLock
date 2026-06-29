# chat_agent.py
# Allows a patient to ask questions about their health documents.
# Uses ChromaDB for local vector storage and RAG (Retrieval Augmented Generation)
# to ground answers in the actual document content.
# Nothing leaves your machine.

import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# --- Configuration ---
MODEL_NAME = "llama3"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"

PROMPT_TEMPLATE = """You are a helpful medical document assistant.
Answer the patient's question using ONLY the information in the documents provided below.
Use plain, simple language a non-medical person can understand.
If the answer is not in the documents, say "I couldn't find that information in your documents."
Never guess or make up medical information.

Documents:
{context}

Patient's question: {question}

Answer:"""


def load_and_chunk_document(file_path: str) -> list:
    """Load a document and split it into chunks for embedding."""
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

    # Split into chunks — small enough to embed, large enough for context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(pages)
    print(f"Document split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(file_path: str) -> Chroma:
    """Embed document chunks and store in ChromaDB."""
    print(f"\nLoading and embedding: {file_path}")
    chunks = load_and_chunk_document(file_path)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("Building vector store — this may take a moment...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store ready.")
    return vectorstore


def ask_question(vectorstore: Chroma, question: str) -> str:
    """Retrieve relevant chunks and generate a grounded answer."""
    # Find the 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_chunks = retriever.invoke(question)

    if not relevant_chunks:
        return "I couldn't find any relevant information in your documents."

    # Build context from retrieved chunks
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    # Show sources for transparency
    print(f"\nRetrieved {len(relevant_chunks)} relevant chunks.")

    llm = OllamaLLM(model=MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = llm.invoke(prompt).strip()

    return answer


def chat_session(file_path: str):
    """Run an interactive chat session about a document."""
    vectorstore = build_vector_store(file_path)

    print("\n--- CHAT SESSION ---")
    print("Ask questions about your document. Type 'quit' to exit.\n")

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