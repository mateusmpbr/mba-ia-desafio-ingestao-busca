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
        raise RuntimeError(f"PDF não encontrado em {pdf_path}")

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("URL de conexão com o banco de dados não definida. Configure DATABASE_URL no .env")

    collection = os.getenv("PG_VECTOR_COLLECTION_NAME") or "documents"

    docs = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False)
    splits = splitter.split_documents(docs)

    if not splits:
        print("Nenhum chunk foi gerado a partir do documento. Nada a ingerir.")
        return

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]

    ids = [f"doc-{i}-{collection}" for i in range(len(enriched))]

    googleApiKey = os.getenv("GOOGLE_API_KEY")
    openaiApiKey = os.getenv("OPENAI_API_KEY")

    if(not googleApiKey and not openaiApiKey):
        raise RuntimeError("Chave de API não encontrada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no .env")

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
    print(f"Ingeridos {len(enriched)} chunks na coleção '{collection}'")


if __name__ == "__main__":
    ingest_pdf()