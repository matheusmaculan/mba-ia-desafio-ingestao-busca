# ============================================================
# ingest.py — PARTE 1: INGESTÃO DO PDF
# ============================================================
# Fluxo desta etapa (roda UMA vez, antes de abrir o chat):
#
#   document.pdf
#      -> PyPDFLoader              (lê o texto do PDF)
#      -> RecursiveCharacterTextSplitter  (quebra em pedaços)
#      -> Embeddings               (transforma texto em vetor)
#      -> PGVector                 (salva os vetores no Postgres)
#
# Depois de rodar este script, os pedaços do PDF ficam guardados
# no banco, prontos para a BUSCA (search.py / chat.py).
# ============================================================

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document


load_dotenv()


def _criar_embeddings():
    """Cria o modelo de embeddings conforme a chave de API disponível no .env.

    Embeddings convertem texto em vetores de números. Textos parecidos
    ficam com vetores parecidos — é isso que permite a busca semântica.

    Suporta OpenAI ou Google (Gemini). A escolha é feita pela chave
    que estiver preenchida no .env.
    """
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    if os.getenv("GOOGLE_API_KEY"):
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        )

    raise RuntimeError(
        "Nenhuma chave de API configurada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env."
    )


# 10 chunks de ~1000 caracteres ficam bem abaixo dos 30 mil tokens/minuto
# do plano gratuito. A pausa evita empilhar vários lotes na mesma janela.
_TAMANHO_LOTE = 10
_PAUSA_ENTRE_LOTES_S = 20
_TENTATIVAS_COTA = 5


def _segundos_de_espera(erro: BaseException) -> float:
    """Lê o 'retry in 19.3s' que o Gemini devolve no 429."""
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(erro), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2
    return 60.0


def _salvar_em_lotes(store: PGVector, documentos: list[Document], ids: list[str]) -> None:
    total = len(documentos)
    for inicio in range(0, total, _TAMANHO_LOTE):
        fim = min(inicio + _TAMANHO_LOTE, total)
        lote = documentos[inicio:fim]
        lote_ids = ids[inicio:fim]
        print(f"      lote {inicio + 1}-{fim} de {total}")

        for tentativa in range(1, _TENTATIVAS_COTA + 1):
            try:
                store.add_documents(documents=lote, ids=lote_ids)
                break
            except Exception as erro:
                mensagem = str(erro)
                cota_estourada = "429" in mensagem or "ResourceExhausted" in mensagem
                if not cota_estourada or tentativa == _TENTATIVAS_COTA:
                    raise
                espera = _segundos_de_espera(erro)
                print(
                    f"      cota do Gemini estourada, esperando {espera:.0f}s "
                    f"(tentativa {tentativa}/{_TENTATIVAS_COTA})..."
                )
                time.sleep(espera)

        if fim < total:
            time.sleep(_PAUSA_ENTRE_LOTES_S)


def ingest_pdf() -> None:
    """Lê o PDF, divide em chunks, gera embeddings e salva no Postgres/pgVector."""

    database_url = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    pdf_path = os.getenv("PDF_PATH")

    if not database_url:
        raise RuntimeError("DATABASE_URL não está definida no .env.")
    if not collection_name:
        raise RuntimeError("PG_VECTOR_COLLECTION_NAME não está definida no .env.")
    if not pdf_path:
        pdf_path = Path(__file__).resolve().parent.parent / "document.pdf"
    else:
        pdf_path = Path(pdf_path)

    print(f"[1/4] Carregando PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    if not docs:
        raise SystemExit("PDF vazio ou não encontrado. Nada para ingerir.")

    print("[2/4] Dividindo o PDF em chunks de 1000 chars (overlap 150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise SystemExit("Nenhum chunk foi gerado. Verifique o conteúdo do PDF.")
    print(f"      {len(chunks)} chunks gerados.")

    docs_limpos = [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
    ]

    ids = [f"doc-{i}" for i in range(len(docs_limpos))]

    print("[3/4] Conectando ao Postgres/pgVector e gerando embeddings...")
    embeddings = _criar_embeddings()

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )
    
    print("[4/4] Salvando os chunks no banco...")
    _salvar_em_lotes(store, docs_limpos, ids)

    print(f"Ingestão concluída: {len(docs_limpos)} chunks salvos na collection '{collection_name}'.")


if __name__ == "__main__":
    ingest_pdf()