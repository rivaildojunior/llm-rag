import os
import re
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "rag_test"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# Schema das tabelas disponíveis para o LLM gerar SQL
DB_SCHEMA = """
Tabela: orders
Colunas: id (integer), customer_name (text), status (text)
Valores possíveis de status: PENDENTE, ENVIADO, ENTREGUE
"""


class DbService:

    # Inicializa o cliente OpenAI e estabelece a conexão com o banco de dados PostgreSQL
    def __init__(self):
        self.client = OpenAI()
        self.conn = psycopg2.connect(**DB_CONFIG)

    # Usa o LLM para converter uma pergunta em linguagem natural em uma query SQL
    def _generate_sql(self, query: str) -> str:
        prompt = f"""Você é um assistente que converte perguntas em SQL para PostgreSQL.
                    Use apenas a tabela descrita abaixo. Retorne SOMENTE o SQL, sem explicações, sem markdown.
                    Para comparações de texto (nomes, status), use sempre ILIKE para ignorar maiúsculas/minúsculas.

                    Schema:
                    {DB_SCHEMA}

                    Pergunta: {query}
                    SQL:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    # Recebe uma pergunta, gera o SQL correspondente, executa no banco e retorna o resultado formatado
    def query(self, query: str) -> str:
        sql = self._generate_sql(query)

        # Remove blocos markdown que o LLM pode incluir (```sql ... ```)
        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()

        # Bloqueia qualquer operação que não seja SELECT
        if not sql.lower().startswith("select"):
            return "Apenas consultas SELECT são permitidas."

        with self.conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        if not rows:
            return "Nenhum resultado encontrado no banco de dados."

        # Formata os dados brutos e passa para o LLM gerar uma resposta em linguagem natural
        lines = [", ".join(columns)]
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
        data_text = "\n".join(lines)

        return self._naturalize(query, data_text)

    # Converte o resultado tabular em uma resposta amigável em linguagem natural
    def _naturalize(self, question: str, data: str) -> str:
        prompt = f"""Você é um assistente de atendimento ao cliente. Com base nos dados abaixo, 
                responda à pergunta do usuário de forma clara e amigável, em português.
                Não mencione SQL, tabelas ou dados técnicos.

                Dados:
                {data}

                Pergunta: {question}
                Resposta:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
