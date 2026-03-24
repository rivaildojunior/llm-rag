from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import time  

# Carrega variáveis de ambiente (ex: OPENAI_API_KEY)
load_dotenv()

# 🔹 Configuração global do LlamaIndex
# Define o LLM que será usado para gerar as respostas
Settings.llm = OpenAI(model="gpt-5", temperature=0.1)

# Define o modelo de embeddings usado para indexação e busca semântica
Settings.embed_model = OpenAIEmbedding()

# 🔹 Leitura da base de conhecimento externa
# Todo o conteúdo do arquivo será a "fonte de verdade" do RAG
with open("dados.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 🔹 Transformação do texto em documentos
# Cada parágrafo vira um Document, que é a unidade básica do LlamaIndex
documents = [
    Document(text=chunk.strip())
    for chunk in content.split("\n\n")
    if chunk.strip()
]

# 🔹 Criação do índice vetorial
# Aqui o LlamaIndex:
# - gera embeddings dos documentos
# - armazena os vetores
# - prepara a base para recuperação semântica (RAG)
index = VectorStoreIndex.from_documents(documents)

# 🔹 Query engine
# Responsável por:
# - receber a pergunta do usuário
# - recuperar documentos relevantes do índice
# - sintetizar a resposta com base apenas nesses documentos
query_engine = index.as_query_engine(similarity_top_k=6)

# 🔹 Memória simples em texto
# Guarda o histórico da conversa para permitir resolver referências como "ela"
chat_history = []

print("🤖 Chat iniciado com RAG + memória simples (LlamaIndex + OpenAI).\n")

while True:
    user_input = input("Você: ")

    # Condição de saída do chat
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("👋 Encerrando o chat.")
        break

    # 🔹 Construção da pergunta com memória
    # As últimas interações são concatenadas à pergunta atual
    # para dar contexto ao LLM (memória conversacional básica)
    if chat_history:
        history_context = "\n".join(chat_history[-4:])  # últimas interações
        query = f"CONVERSA ANTERIOR:\n{history_context}\n\nPERGUNTA:\n{user_input}"
    else:
        query = user_input

    # 🔹 Executa a consulta RAG
    response = query_engine.query(query)

    # 🔹 Regra de RAG puro
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


    # 🔹 Atualiza a memória da conversa
    chat_history.append(f"Usuário: {user_input}")
    chat_history.append(f"Assistente: {answer}")
