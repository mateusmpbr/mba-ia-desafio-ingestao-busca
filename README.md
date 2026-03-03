# Desafio MBA Engenharia de Software com IA - Full Cycle

Este repositório implementa ingestão de um PDF em um banco PostgreSQL com pgVector e uma interface de busca via CLI.

## Pré-requisitos

- Docker e Docker Compose
- Linux
- Python 3.11 (se o binário for diferente ajuste `./setup.sh`, por exemplo `python3` ou `python3.12`, só é necessário ser >= 3.11)

## Passos de instalação e execução

### 1. Preparar o ambiente e dependências

```bash
./setup.sh
```

O script `setup.sh` copia `.env.example` para `.env`, cria um `venv` em `./venv` e instala dependências.

### 2. Preencher o arquivo `.env`

Após o `setup.sh` será gerado o arquivo `.env`. Preencha pelo menos uma das chaves de API:

- `OPENAI_API_KEY` — usa modelos e embeddings da OpenAI
- `GOOGLE_API_KEY` — usa modelos e embeddings do Google (Gemini)

Regras de seleção de modelo

- Se `GOOGLE_API_KEY` estiver preenchido, o código usa embeddings/LLM do Google.
- Se `GOOGLE_API_KEY` não estiver preenchido, mas `OPENAI_API_KEY` estiver, o código usa OpenAI.
- Se nenhuma estiver preenchida, o processo de ingestão/busca lançará erro.

### 3. Subir o banco de dados (Postgres + pgVector)

```bash
docker compose up -d
```

### 4. Ativar o ambiente virtual

```bash
source venv/bin/activate
```

### 5. Ingestão do PDF

```bash
python src/ingest.py
```

O script irá:

- Ler o arquivo definido em `PDF_PATH` dentro do `.env` (ou `document.pdf` por padrão).
- Dividir o PDF em chunks de 1000 caracteres com overlap de 150.
- Gerar embeddings (OpenAI ou Google, conforme o `.env`).
- Salvar os vetores no Postgres/pgVector na coleção definida por `PG_VECTOR_COLLECTION_NAME` (ou `documents` por padrão).

### 6. Executar o chat CLI

```bash
python src/chat.py
```

No CLI digite sua pergunta e pressione Enter. Digite `sair` para encerrar.

Observação importante sobre coleções

Se você executar ingestão com o modelo do Google e depois quiser executar com OpenAI, altere `PG_VECTOR_COLLECTION_NAME` no `.env` para um nome diferente antes de rodar a ingestão novamente. Isso evita misturar vetores embeddados por modelos diferentes na mesma coleção.

## Variáveis úteis no `.env`

- `DATABASE_URL` ou `PGVECTOR_URL`: string de conexão com o Postgres (ex.: `postgresql://postgres:postgres@localhost:5432/rag`)
- `PDF_PATH`: caminho para o PDF a ser ingerido (padrão `document.pdf`)
- `PG_VECTOR_COLLECTION_NAME`: nome da coleção no pgVector (padrão `documents`)
- `OPENAI_EMBEDDING_MODEL`, `OPENAI_RESPONSE_MODEL` — modelos OpenAI
- `GOOGLE_EMBEDDING_MODEL`, `GOOGLE_RESPONSE_MODEL` — modelos Google

## Execução rápida (resumo dos comandos)

```bash
./setup.sh
# editar .env (preencher OPENAI_API_KEY ou GOOGLE_API_KEY)
docker compose up -d
source venv/bin/activate
python src/ingest.py
python src/chat.py
```

## Logs e mensagens

As mensagens, erros e prompts do CLI estão em português.

## Suporte

Se houver problemas de import de pacotes, verifique se o `venv` está ativado e se as dependências foram instaladas com sucesso (`pip install -r requirements.txt`).
