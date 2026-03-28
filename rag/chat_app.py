import time
from rag.rag_service import RagService
from rag.memory_service import MemoryService
from rag.guardrails_service import GuardrailsService


class ChatApp:
    def __init__(self):
        self.rag = RagService()
        self.memory = MemoryService()
        self.guardrails = GuardrailsService()

    def run(self):
        print("🤖 Chat iniciado com RAG + memória simples (LlamaIndex + OpenAI).\n")

        while True:
            user_input = input("Você: ")

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando o chat.")
                break

            # Validar input
            is_valid, error_msg = self.guardrails.validate_input(user_input)
            if not is_valid:
                print(f"🤖 Assistente: {error_msg}\n")
                continue

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

            # Validar resposta
            is_response_valid, warning_msg = self.guardrails.validate_response(response)
            
            if response.source_nodes and is_response_valid:
                answer = response.response
            else:
                answer = warning_msg if warning_msg else "Não sei."

            print("🤖 Assistente: ", end="")
            for char in answer:
                print(char, end="", flush=True)
                time.sleep(0.01)
            print("\n")

            self.memory.add("user", user_input)
            self.memory.add("assistant", answer)