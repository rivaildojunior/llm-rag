from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
import time  

load_dotenv()

MAX_HISTORY_CHARS = 2000

Settings.llm = OpenAI(model="gpt-5", temperature=0.1)

Settings.embed_model = OpenAIEmbedding()

with open("dados.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Configuração de chunking
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=500,      
    chunk_overlap=100    
)

documents = [Document(text=content)]

nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)

# Query engine
# Responsável por:
# - receber a pergunta do usuário
# - recuperar documentos relevantes do índice
# - sintetizar a resposta com base apenas nesses documentos
query_engine = index.as_query_engine(similarity_top_k=6)

# Guarda o histórico da conversa para permitir resolver referências como "ela"
chat_history = []

print("🤖 Chat iniciado com RAG + memória simples (LlamaIndex + OpenAI).\n")

def trim_chat_history(chat_history, max_chars):
    total = 0
    trimmed = []

    # percorre de trás pra frente (mensagens mais recentes primeiro)
    for msg in reversed(chat_history):
        msg_len = len(msg["content"])
        if total + msg_len > max_chars:
            break
        trimmed.append(msg)
        total += msg_len

    return list(reversed(trimmed))

while True:
    user_input = input("Você: ")

    if user_input.lower() in ["sair", "exit", "quit"]:
        print("👋 Encerrando o chat.")
        break

    # As últimas interações são concatenadas à pergunta atual
    # para dar contexto ao LLM (memória conversacional básica)
    trimmed_history = trim_chat_history(chat_history, MAX_HISTORY_CHARS)

    history_context = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in trimmed_history]
    )     

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
    # Executa a consulta RAG
    response = query_engine.query(query)

    # Só responde se algum documento relevante foi recuperado
    if response.source_nodes:
        answer = response.response
    else:
        answer = "Não sei."

    print("🤖 Assistente: ", end="")
    for char in answer:
        print(char, end="", flush=True)
        time.sleep(0.01)
    print("\n")


    # Atualiza a memória da conversa
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})
