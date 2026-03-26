import time
from rag.rag_service import RagService
from rag.memory_service import MemoryService


class ChatApp:
    def __init__(self):
        self.rag = RagService()
        self.memory = MemoryService()

    def run(self):
        print("🤖 Chat iniciado com RAG + memória simples (LlamaIndex + OpenAI).\n")

        while True:
            user_input = input("Você: ")

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando o chat.")
                break

            history_context = self.memory.get_context()

            query = f"""
            Você é um assistente que prioriza responder com base nos documentos fornecidos.

            Regras:
            - Use os documentos como principal fonte de verdade
            - Se a informação não estiver explícita, diga que não encontrou no conteúdo
            - Evite inventar informações
            - Seja claro e amigável

            CONVERSA ANTERIOR:
            {history_context}

            PERGUNTA:
            {user_input}
            """

            response = self.rag.query(query)

            if response.source_nodes:
                answer = response.response
            else:
                answer = "Não sei."

            print("🤖 Assistente: ", end="")
            for char in answer:
                print(char, end="", flush=True)
                time.sleep(0.01)
            print("\n")

            self.memory.add("user", user_input)
            self.memory.add("assistant", answer)