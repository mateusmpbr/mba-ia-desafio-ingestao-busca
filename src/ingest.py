import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()


def ingest_pdf():
    # Resolve env vars with some fallbacks to support multiple templates
    pdf_path = os.getenv("PDF_PATH") or "document.pdf"
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found at {pdf_path}")

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("Database connection URL not set. Set DATABASE_URL or PGVECTOR_URL in .env")

    collection = os.getenv("PG_VECTOR_COLLECTION_NAME") or "documents"

    docs = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False)
    splits = splitter.split_documents(docs)

    if not splits:
        print("No document splits produced. Nothing to ingest.")
        return

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    googleApiKey = os.getenv("GOOGLE_API_KEY")
    openaiApiKey = os.getenv("OPENAI_API_KEY")

    if(not googleApiKey and not openaiApiKey):
        raise RuntimeError("API key not found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env")

    use_google = bool(googleApiKey)

    if use_google:
        embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/gemini-embedding-001")
    else:
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small")

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=db_url,
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)
    print(f"Ingested {len(enriched)} chunks into collection '{collection}'")


if __name__ == "__main__":
    ingest_pdf()