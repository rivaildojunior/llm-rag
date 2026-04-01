from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder


class RagService:
    
    def __init__(self):
        # Configuração do LLM (OpenAI) para geração de texto.
        # Na POC estamos usando o modelo "gpt-5" com temperatura baixa para respostas mais consistentes.
        Settings.llm = OpenAI(model="gpt-5", temperature=0.1)

        # Configuração do modelo de embeddings para conversão de textos em vetores.
        Settings.embed_model = OpenAIEmbedding()

        # Leitura do arquivo local de dados que será indexado.
        # Esse é o corpus de conhecimento para o RAG.
        with open("dados.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # Configuração do parser de nós (chunks) para dividir o texto em pedaços controlados.
        # chunk_size = tamanho aproximado de cada pedaço em tokens/caracteres;
        # chunk_overlap = quanto os pedaços podem se sobrepor para manter contexto.
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=500,
            chunk_overlap=100
        )

        # Cria um documento com o conteúdo carregado e converte em nós.
        documents = [Document(text=content)]
        nodes = node_parser.get_nodes_from_documents(documents)

        # Conecta ao Qdrant (rodando no Docker)
        client = QdrantClient(host="localhost", port=6333)
        vector_store = QdrantVectorStore(client=client, collection_name="rag_collection")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if client.collection_exists("rag_collection"):
            # Carrega índice existente
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        else:
            # Cria novo índice
            index = VectorStoreIndex(nodes, storage_context=storage_context)

        # Cria o mecanismo de consulta (query engine) usando similaridade para retornar hits relevantes.
        # similarity_top_k=10 indica os 10 documentos mais similares para cada consulta (antes do re-ranking).
        self.query_engine = index.as_query_engine(similarity_top_k=10)
        
        # Inicializa o cross-encoder para re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def query(self, query: str):
        # Executa a consulta inicial no mecanismo de RAG
        initial_response = self.query_engine.query(query)
        
        # Se não há source_nodes, retorna resposta vazia
        if not initial_response.source_nodes:
            return initial_response
        
        # Extrai textos dos nodes para re-ranking
        node_texts = [node.text for node in initial_response.source_nodes]
        
        # Cria pares (query, text) para o cross-encoder
        query_text_pairs = [[query, text] for text in node_texts]
        
        # Calcula scores de relevância com cross-encoder
        scores = self.cross_encoder.predict(query_text_pairs)
        
        # Ordena nodes por score (decrescente)
        scored_nodes = list(zip(scores, initial_response.source_nodes))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        # Seleciona apenas os top 3 mais relevantes após re-ranking
        top_nodes = [node for score, node in scored_nodes[:3]]
        
        # Atualiza a resposta com os nodes re-rankados
        initial_response.source_nodes = top_nodes
        
        return initial_response