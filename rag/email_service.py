# Serviço MCP (Model Context Protocol) responsável pelo envio de emails via SMTP.
# Expõe a ferramenta 'send_email' para ser consumida por agentes LLM através do protocolo MCP.

import smtplib
import os
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Carrega variáveis de ambiente do arquivo .env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD)
load_dotenv()

# Instância do servidor MCP identificado como "email-service"
app = Server("email-service")


# Função utilitária que realiza o envio do email diretamente via protocolo SMTP.
# Lê as credenciais e configurações do servidor de email a partir de variáveis de ambiente
# (.env), monta a mensagem no formato MIME e estabelece uma conexão segura com o servidor
# usando STARTTLS antes de autenticar e enviar.
# É chamada internamente pela tool MCP e não é exposta diretamente ao agente LLM.
def send_email(to: str, subject: str, body: str) -> None:
    """Envia um email via SMTP usando as variáveis de ambiente configuradas."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to, msg.as_string())


# Handler do protocolo MCP responsável por anunciar ao cliente quais tools este servidor oferece.
# Quando um agente LLM se conecta ao servidor, ele chama list_tools para descobrir as capacidades
# disponíveis antes de decidir qual invocar. Aqui registramos a tool "send_email" com seu
# esquema JSON Schema, que define os parâmetros esperados e suas descrições.
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declara as ferramentas disponíveis neste servidor MCP."""
    return [
        types.Tool(
            name="send_email",
            description="Envia um email simples via SMTP",
            inputSchema={
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Endereço do destinatário"},
                    "subject": {"type": "string", "description": "Assunto do email"},
                    "body":    {"type": "string", "description": "Corpo do email"},
                },
                "required": ["to", "subject", "body"],
            },
        )
    ]


# Handler do protocolo MCP que recebe e executa a chamada de tool feita pelo agente LLM.
# O MCP roteia todas as invocações de ferramentas para cá, passando o nome da tool e os
# argumentos já desserializados. A função valida o nome recebido e delega a execução real
# para send_email(), devolvendo o resultado como uma lista de TextContent — formato
# padrão de resposta do protocolo MCP.
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Executa a ferramenta solicitada pelo agente LLM."""
    if name != "send_email":
        return [types.TextContent(type="text", text="Tool não encontrada")]

    send_email(
        to=arguments["to"],
        subject=arguments["subject"],
        body=arguments["body"],
    )

    return [types.TextContent(type="text", text=f"Email enviado para {arguments['to']}")]


# Ponto de entrada do servidor MCP. Inicia o servidor usando stdin/stdout como canal de
# comunicação (transporte stdio), que é o padrão esperado quando o processo é iniciado
# como subprocess pelo cliente MCP (chat_app.py). O servidor fica ativo aguardando
# requisições até que o processo pai encerre a conexão.
async def main():
    """Ponto de entrada assíncrono: inicia o servidor MCP via stdin/stdout."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
