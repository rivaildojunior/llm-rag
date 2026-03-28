import streamlit as st
import time
from dotenv import load_dotenv
from rag.rag_service import RagService
from rag.memory_service import MemoryService
from rag.guardrails_service import GuardrailsService

load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="RAG Chat Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: right;
        color: #1976d2;
    }
    .assistant-message {
        background-color: #f3e5f5;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #7b1fa2;
    }
    </style>
""", unsafe_allow_html=True)

# Inicialização da sessão
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RagService()
    st.session_state.memory = MemoryService()
    st.session_state.guardrails = GuardrailsService()
    st.session_state.last_processed_input = None
    st.session_state.last_animated_message_count = 0
    st.session_state.submit_count = 0

# Título
st.title("🤖 Chat com RAG + LLM")
st.markdown("---")

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")

    if st.button("🗑️ Limpar Histórico"):
        st.session_state.memory.chat_history = []
        st.success("Histórico limpo!")

# Exibir histórico de conversa
st.subheader("💬 Conversa")
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.memory.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><b>Você:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Exibir resposta do assistente com efeito de digitação apenas para a última mensagem (uma única vez)
            is_last_message = i == len(st.session_state.memory.chat_history) - 1
            should_animate = is_last_message and st.session_state.last_animated_message_count < len(st.session_state.memory.chat_history)
            
            if should_animate:
                # Última mensagem nova - mostrar com efeito de digitação
                response_placeholder = st.empty()
                displayed_text = ""
                for char in msg["content"]:
                    displayed_text += char
                    response_placeholder.markdown(f'<div class="assistant-message"><b>🤖 Assistente:</b> {displayed_text}</div>', unsafe_allow_html=True)
                    time.sleep(0.01)
                # Marcar que essa mensagem já foi animada
                st.session_state.last_animated_message_count = len(st.session_state.memory.chat_history)
            else:
                # Mensagens anteriores ou já animadas - mostrar normalmente
                st.markdown(f'<div class="assistant-message"><b>🤖 Assistente:</b> {msg["content"]}</div>', unsafe_allow_html=True)

# Input do usuário
st.markdown("---")

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Digite sua pergunta:", placeholder="Faça uma pergunta sobre o conteúdo...", key=f"user_input_{st.session_state.submit_count}")
with col2:
    submit_button = st.button("📤 Enviar", use_container_width=True)

if submit_button and user_input:
    # Validar input
    is_valid, error_msg = st.session_state.guardrails.validate_input(user_input)
    if not is_valid:
        st.error(error_msg)
    # Verificar se a pergunta já foi processada (evita loop)
    elif user_input != st.session_state.last_processed_input:
        st.session_state.last_processed_input = user_input
        
        # Adicionar mensagem do usuário ao histórico
        st.session_state.memory.add("user", user_input)
        
        # Obter contexto da conversa
        history_context = st.session_state.memory.get_context()
        
        # Preparar query para o RAG
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
        
        # Mostrar spinner enquanto processa
        with st.spinner("🔄 Processando sua pergunta..."):
            response = st.session_state.rag_service.query(query)
            
            # Validar resposta
            is_response_valid, warning_msg = st.session_state.guardrails.validate_response(response)
            
            if response.source_nodes and is_response_valid:
                answer = response.response
            else:
                answer = warning_msg if warning_msg else "Desculpe, não encontrei informações sobre isso no conteúdo disponível."
        
        # Adicionar resposta ao histórico
        st.session_state.memory.add("assistant", answer)
        
        # Incrementar contador para limpar o campo
        st.session_state.submit_count += 1
        
        # Recarregar para mostrar a nova mensagem
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    POC de RAG com LLM | Powered by LlamaIndex + OpenAI + Streamlit
</div>
""", unsafe_allow_html=True)
