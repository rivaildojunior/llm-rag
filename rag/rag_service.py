from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response_synthesizers import get_response_synthesizer
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
        
        # Cria um retriever para recuperar 10 nodes antes do re-ranking
        self.retriever = index.as_retriever(similarity_top_k=10)
        
        # Inicializa o cross-encoder para re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Inicializa o sintetizador de respostas
        self.response_synthesizer = get_response_synthesizer()

    def query(self, query: str):
        # RETRIEVAL: Recupera os 10 chunks mais similares usando similarity search
        retrieved_nodes = self.retriever.retrieve(query)
        
        if not retrieved_nodes:
            return None
        
        # RE-RANKING: Usa cross-encoder para reranquear os 10 chunks
        node_texts = [node.text for node in retrieved_nodes]
        query_text_pairs = [[query, text] for text in node_texts]
        scores = self.cross_encoder.predict(query_text_pairs)
        
        # SELEÇÃO: Ordena os nodes pelos scores e seleciona os top 3
        scored_nodes = list(zip(scores, retrieved_nodes))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [node for score, node in scored_nodes[:3]]
        
        # GENERATION: Passa apenas os 3 chunks mais relevantes para o LLM gerar a resposta
        response = self.response_synthesizer.synthesize(query, nodes=top_nodes)
        
        return response