# ============================================================
# chat.py — CLI para interação com o usuário
# ============================================================
# Loop de chat no terminal:
#   - o usuário digita uma pergunta
#   - a pergunta vai para o search_prompt() (busca + LLM)
#   - a resposta é exibida
#   - o loop continua até o usuário digitar "sair"/"exit"
# ============================================================

from search import search_prompt


def main():
    print("=" * 60)
    print(" Chat com o PDF — digite sua pergunta")
    print(" Digite 'sair' ou 'exit' para encerrar.")
    print("=" * 60)
    print()

    while True:
        pergunta = input("\nPERGUNTA: ").strip()

        if pergunta.lower() in ("sair", "exit", "quit", "q"):
            print("Encerrando o chat. Até mais!")
            break

        if not pergunta:
            continue

        try:
            resposta = search_prompt(pergunta)
        except Exception as erro:
            resposta = f"Erro ao processar sua pergunta: {erro}"

        print(f"RESPOSTA: {resposta}")


if __name__ == "__main__":
    main()
