PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
  import os
  from dotenv import load_dotenv
  from langchain_openai import OpenAI
  from langchain_openai import OpenAIEmbeddings
  from langchain_postgres import PGVector

  load_dotenv()

  if not question:
    return None

  db_url = os.getenv("DATABASE_URL")

  if not db_url:
    raise RuntimeError("Database connection URL not set. Set DATABASE_URL or PGVECTOR_URL in .env")

  collection = os.getenv("PG_VECTOR_COLLECTION_NAME") or "documents"

  embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small")

  store = PGVector(
    embeddings=embeddings,
    collection_name=collection,
    connection=db_url,
    use_jsonb=True,
  )

  results = store.similarity_search_with_score(question, k=10)

  contexto_parts = []
  for doc, _ in results:
    texto = doc.page_content.strip()
    if texto:
      contexto_parts.append(texto)

  contexto = "\n\n---\n\n".join(contexto_parts)

  prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)

  # If no context, follow the instructions and return the fallback message
  if not contexto.strip():
    return "Não tenho informações necessárias para responder sua pergunta."

  # Call LLM to answer using only the provided prompt
  llm = OpenAI(model=os.getenv("OPENAI_RESPONSE_MODEL") or "gpt-5-nano", temperature=0)

  response = llm(prompt)

  # OpenAI client returns an object; try to extract text
  if isinstance(response, str):
    return response.strip()

  # Attempt to read common attribute
  text = getattr(response, "content", None) or getattr(response, "text", None)
  if text:
    return text.strip()

  return str(response)