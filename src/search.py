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
  from langchain_openai import ChatOpenAI
  from langchain_openai import OpenAIEmbeddings
  from langchain_google_genai import GoogleGenerativeAI
  from langchain_google_genai import GoogleGenerativeAIEmbeddings
  from langchain_postgres import PGVector

  load_dotenv()

  if not question:
    return None

  db_url = os.getenv("DATABASE_URL")

  if not db_url:
    raise RuntimeError("URL de conexão com o banco de dados não definida. Configure DATABASE_URL no .env")

  collection = os.getenv("PG_VECTOR_COLLECTION_NAME") or "documents"

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
  if use_google:
    llm = GoogleGenerativeAI(model=os.getenv("GOOGLE_RESPONSE_MODEL") or "gemini-2.5-flash-lite", temperature=0)
  else:
    llm = ChatOpenAI(model=os.getenv("OPENAI_RESPONSE_MODEL") or "gpt-5-nano", temperature=0)

  response = llm.invoke(prompt)

  # OpenAI client returns an object; try to extract text
  if isinstance(response, str):
    return response.strip()

  # Attempt to read common attribute
  text = getattr(response, "content", None) or getattr(response, "text", None)
  if text:
    return text.strip()

  return str(response)