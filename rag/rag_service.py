from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser


class RagService:
    def __init__(self):
        Settings.llm = OpenAI(model="gpt-5", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding()

        with open("dados.txt", "r", encoding="utf-8") as f:
            content = f.read()

        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=500,
            chunk_overlap=100
        )

        documents = [Document(text=content)]
        nodes = node_parser.get_nodes_from_documents(documents)

        index = VectorStoreIndex(nodes)

        self.query_engine = index.as_query_engine(similarity_top_k=6)

    def query(self, query: str):
        return self.query_engine.query(query)