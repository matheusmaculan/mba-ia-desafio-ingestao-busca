# Ingestão e Busca Semântica com LangChain e Postgres (RAG)

Sistema de **RAG (Retrieval-Augmented Generation)** que lê um PDF, guarda seus
trechos como vetores em um PostgreSQL com pgVector, e responde perguntas do
usuário via linha de comando usando **apenas** o conteúdo do PDF.

## Como funciona

```
=== Ingestão (roda 1 vez) ===
document.pdf
   → PyPDFLoader                (lê o texto)
   → RecursiveCharacterTextSplitter  (quebra em chunks de 1000 chars, overlap 150)
   → Embeddings (Gemini)        (texto → vetor)
   → PGVector                   (salva no Postgres/pgVector)

=== Consulta (a cada pergunta) ===
Pergunta
   → Embeddings                 (pergunta → vetor)
   → similarity_search (k=10)   (busca os 10 trechos mais parecidos)
   → Prompt (CONTEXTO + REGRAS + pergunta)
   → LLM (Gemini)               (gera a resposta presa ao contexto)
```

A regra central: a LLM **não pode inventar**. Se a informação não estiver nos
trechos encontrados, ela responde:

```
Não tenho informações necessárias para responder sua pergunta.
```

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma API Key do Google (Gemini)

## Configuração

### 1. Subir o banco (PostgreSQL + pgVector)

```bash
docker compose up -d
```

Isso sobe um PostgreSQL com a extensão `vector` já ativada
(imagem `pgvector/pgvector:pg17`), com usuário/senha `postgres/postgres`,
banco `rag`, na porta `5432`.

### 2. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar o `.env`

Copie o `.env.example` para `.env` e preencha:

```bash
cp .env.example .env
```

Variáveis necessárias:

| Variável | Descrição |
|---|---|
| `GOOGLE_API_KEY` | Chave da API do Google (Gemini) |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings (padrão: `models/gemini-embedding-001`) |
| `GOOGLE_LLM_MODEL` | Modelo de LLM para responder (padrão: `gemini-3.5-flash-lite`) |
| `DATABASE_URL` | URL de conexão do Postgres |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection onde os vetores são salvos |
| `PDF_PATH` | Caminho do PDF a ser ingerido (padrão: `document.pdf`) |

Exemplo de `DATABASE_URL` (bate com o docker-compose):

```
postgresql://postgres:postgres@localhost:5432/rag
```

## Execução

### 1. Ingerir o PDF

```bash
python src/ingest.py
```

### 2. Abrir o chat

```bash
python src/chat.py
```

Para sair, digite `sair`, `exit`, `quit` ou `q`.

## Observação: adaptação para cota gratuita do Gemini

O plano gratuito do Gemini limita os embeddings a ~30 mil tokens por minuto.
O `ingest.py` foi adaptado para esse cenário:

- Os chunks são gravados em **lotes pequenos** (10 por vez);
- Entre os lotes há uma **pausa** para não estourar a cota;
- Em caso de `429` (cota estourada), há **retry** com espera calculada a partir
  do erro retornado pelo Gemini.

Com uma cota maior, basta remover o `_salvar_em_lotes(...)` e usar diretamente:

```python
store.add_documents(documents=docs_limpos, ids=ids)
```

## Estrutura do projeto

```
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt        # Dependências
├── .env.example            # Template das variáveis de ambiente
├── document.pdf            # PDF para ingestão
├── src/
│   ├── ingest.py           # Script de ingestão do PDF
│   ├── search.py           # Busca semântica + prompt + LLM
│   └── chat.py             # CLI para interação com o usuário
└── README.md
```
