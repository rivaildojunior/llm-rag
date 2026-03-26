from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser


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

        # Cria o índice de vetor (VectorStoreIndex) a partir dos nós.
        index = VectorStoreIndex(nodes)

        # Cria o mecanismo de consulta (query engine) usando similaridade para retornar hits relevantes.
        # similarity_top_k=6 indica os 6 documentos mais similares para cada consulta.
        self.query_engine = index.as_query_engine(similarity_top_k=6)

    def query(self, query: str):
        # Executa a consulta no mecanismo de RAG e retorna o resultado.
        return self.query_engine.query(query)