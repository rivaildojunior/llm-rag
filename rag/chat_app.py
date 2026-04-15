import time
import os
import asyncio
import json
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rag.rag_service import RagService
from rag.memory_service import MemoryService
from rag.guardrails_service import GuardrailsService

load_dotenv()

# Definição da ferramenta (tool) que será oferecida ao LLM.
# O OpenAI usa esse esquema para saber QUANDO e COM QUAIS argumentos chamar a função.
# O formato segue o padrão JSON Schema: cada campo descreve nome, tipo e propósito do parâmetro.
# Quando o LLM decidir enviar um email, ele preencherá automaticamente "to", "subject" e "body".
_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Envia um email com o resumo ou histórico da conversa",
        "parameters": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "description": "Endereço do destinatário"},
                "subject": {"type": "string", "description": "Assunto do email"},
                "body":    {"type": "string", "description": "Corpo do email"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}


# Função responsável por enviar um email utilizando o protocolo MCP (Model Context Protocol).
# Em vez de chamar uma biblioteca de email diretamente, ela age como cliente MCP:
# inicia um servidor MCP em subprocess (email_service.py), abre uma sessão de comunicação
# via stdio e invoca remotamente a tool "send_email" nesse servidor.
# Essa abordagem desacopla a lógica de envio do chat, permitindo que o serviço de email
# seja substituído ou evoluído de forma independente.
async def _mcp_send_email(to: str, subject: str, body: str) -> str:
    """Chama a tool send_email via servidor MCP."""

    # Define como o servidor MCP será iniciado: via subprocess do Python executando email_service.py.
    # O protocolo MCP usa stdio (stdin/stdout) como canal de comunicação entre cliente e servidor.
    server_params = StdioServerParameters(command="python", args=["rag/email_service.py"])

    # Abre a conexão com o servidor MCP. `read` e `write` são os streams de comunicação.
    # O bloco `async with` garante que o processo filho seja encerrado ao sair.
    async with stdio_client(server_params) as (read, write):

        # Cria uma sessão MCP sobre os streams abertos acima.
        # A sessão gerencia o protocolo de handshake e serialização das mensagens.
        async with ClientSession(read, write) as session:

            # Realiza o handshake inicial do protocolo MCP (troca de capacidades).
            await session.initialize()

            # Invoca remotamente a tool "send_email" no servidor, passando os argumentos.
            # O servidor executa o envio e devolve o resultado como lista de conteúdo.
            result = await session.call_tool("send_email", {"to": to, "subject": subject, "body": body})

            # Retorna o texto da primeira resposta, ou uma mensagem padrão caso esteja vazia.
            return result.content[0].text if result.content else "Email enviado"


class ChatApp:
    def __init__(self):
        self.rag = RagService()
        self.memory = MemoryService()
        self.guardrails = GuardrailsService()
        self._openai = OpenAIClient()

    def _try_tool_call(self, user_input: str, history: list[dict]) -> str | None:
        """Verifica se o LLM quer usar uma tool. Retorna a resposta ou None."""

        # Monta o histórico completo para o LLM ter contexto ao decidir usar ou não a tool.
        # A instrução de sistema restringe o uso de send_email apenas a pedidos explícitos do usuário.
        messages = [
            {"role": "system", "content": "Você é um assistente. Use a ferramenta send_email apenas quando o usuário pedir explicitamente para enviar um email."},
            *history,
            {"role": "user", "content": user_input},
        ]

        # Envia a requisição ao LLM com a lista de tools disponíveis.
        # `tool_choice="auto"` permite que o LLM decida sozinho se deve chamar uma tool ou responder normalmente.
        response = self._openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[_EMAIL_TOOL],
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Se o LLM não decidiu chamar nenhuma tool, retorna None para seguir o fluxo normal do chat.
        if not msg.tool_calls:
            return None

        # Extrai os argumentos preenchidos pelo LLM para a tool (to, subject, body).
        # O LLM devolve os argumentos como string JSON, então é necessário desserializar.
        args = json.loads(msg.tool_calls[0].function.arguments)

        # Executa a corrotina MCP de forma síncrona usando asyncio.run,
        # já que _try_tool_call não é assíncrona mas _mcp_send_email é.
        return asyncio.run(_mcp_send_email(**args))

    def run(self):
        print("🤖 Chat iniciado com RAG + memória simples (LlamaIndex + OpenAI).\n")
        conversation_history = []

        while True:
            user_input = input("Você: ")

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando o chat.")
                break

            # Verifica se o usuário pediu envio de email antes de processar a pergunta normalmente.
            # Caso o LLM identifique essa intenção, a tool MCP é invocada e o resultado é exibido diretamente,
            # pulando toda a pipeline de RAG + guardrails (já que não é uma consulta a documentos).
            tool_response = self._try_tool_call(user_input, conversation_history)
            if tool_response:
                print(f"🤖 Assistente: {tool_response}\n")
                self.memory.add("user", user_input)
                self.memory.add("assistant", tool_response)
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": tool_response})
                continue

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
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})