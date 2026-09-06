import re
import time
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response_synthesizers import get_response_synthesizer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from rag.db_service import DbService


class RagService:
    
    def __init__(self):
        # Configuração do LLM (OpenAI) para geração de texto.
        # gpt-5/gpt-5-mini são modelos de raciocínio (geram tokens de "pensamento" antes de responder,
        # o que era o maior gargalo da consulta). Trocado para gpt-4o-mini, que responde direto e é
        # bem mais rápido para esse caso de Q&A sobre documento (mesmo modelo já usado no db_service.py).
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

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
            chunk_size=300,
            chunk_overlap=100
        )

        # Cria um documento com o conteúdo carregado e converte em nós.
        documents = [Document(text=content)]
        nodes = node_parser.get_nodes_from_documents(documents)

        # Conecta ao Qdrant (rodando no Docker)
        # client = QdrantClient(host="localhost", port=6333)

        # Conecta ao Qdrant em modo embutido (sem precisar de servidor/Docker), persistindo em disco
        client = QdrantClient(path="rag/qdrant_data")
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

        # Padrão de número de pedido (ex: PED-1001). Roteia pro banco só quando a mensagem contém
        # um número desse formato — evita que perguntas de política que mencionam "pedido" (ex:
        # "rastrear meu pedido") sejam desviadas do RAG, e permite responder só com o número
        # (ex: "PED-1002") em mensagens de acompanhamento.
        self._order_number_pattern = re.compile(r"\b[a-z]{2,6}-?\d{3,}\b", re.IGNORECASE)
        self.db_service = DbService()

    def query(self, query: str, user_input: str = None):
        # Roteamento baseado apenas no input original do usuário, não no prompt completo com histórico.
        # Um número de pedido (ex: PED-1002) já é um sinal suficiente por si só — permite que o
        # usuário responda só com o número, sem repetir "pedido"/"status" na mensagem.
        routing_text = (user_input or query).lower()
        has_order_number = bool(self._order_number_pattern.search(routing_text))
        if has_order_number:
            return self.db_service.query(user_input or query)

        # Usa a pergunta "crua" do usuário (sem o prompt de instruções/histórico) para
        # retrieval e rerank — o texto extra do prompt completo diluía o embedding de busca
        # e fazia chunks relevantes saírem do top-k.
        retrieval_query = user_input or query

        # RETRIEVAL: Recupera os 10 chunks mais similares usando similarity search
        _t0 = time.perf_counter()
        retrieved_nodes = self.retriever.retrieve(retrieval_query)
        print(f"[TIMING] retrieval: {time.perf_counter() - _t0:.2f}s")

        if not retrieved_nodes:
            return None

        # RE-RANKING: Usa cross-encoder para reranquear os 10 chunks
        _t1 = time.perf_counter()
        node_texts = [node.text for node in retrieved_nodes]
        query_text_pairs = [[retrieval_query, text] for text in node_texts]
        scores = self.cross_encoder.predict(query_text_pairs)
        print(f"[TIMING] rerank: {time.perf_counter() - _t1:.2f}s")

        # SELEÇÃO: Ordena os nodes pelos scores e seleciona os top 3
        scored_nodes = list(zip(scores, retrieved_nodes))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [node for score, node in scored_nodes[:3]]

        # GENERATION: Passa apenas os 3 chunks mais relevantes para o LLM gerar a resposta
        _t2 = time.perf_counter()
        response = self.response_synthesizer.synthesize(query, nodes=top_nodes)
        print(f"[TIMING] synthesize (LLM): {time.perf_counter() - _t2:.2f}s")

        return response