# ============================================================
# search.py — PARTE 2: BUSCA SEMÂNTICA + GERAÇÃO DE RESPOSTA
# ============================================================
# Fluxo desta etapa (roda A CADA pergunta do usuário):
#
#   pergunta
#      -> Embeddings            (vira vetor)
#      -> similarity_search     (busca os 10 trechos mais parecidos)
#      -> Prompt (CONTEXTO + REGRAS + pergunta)
#      -> LLM                   (gera a resposta)
#
# A chave do RAG: a LLM responde SÓ com base no CONTEXTO encontrado.
# Se a informação não estiver nos trechos, ela NÃO pode inventar.
# ============================================================

import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

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


def _criar_vector_store() -> PGVector:
    """Cria a conexão com o Postgres/pgVector (mesma config da ingestão)."""
    return PGVector(
        embeddings=GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        ),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def search_prompt(question: str):
    """Busca os trechos relevantes no banco e gera a resposta com a LLM.

    Passos:
      1. Conecta no banco vetorial.
      2. Busca os 10 chunks mais similares à pergunta (k=10).
      3. Junta esses chunks no CONTEXTO.
      4. Monta o prompt e chama a LLM.
      5. Retorna o texto da resposta.
    """
    store = _criar_vector_store()
    resultados = store.similarity_search_with_score(question, k=10)

    if not resultados:
        return "Não tenho informações necessárias para responder sua pergunta."

    contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite"),
        temperature=0,
        timeout=30,
    )

    resposta = llm.invoke(prompt)
    return resposta.content
