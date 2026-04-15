# eval_service.py — Avaliação objetiva do pipeline RAG usando a biblioteca RAGAS.
#
# O RAGAS mede três aspectos fundamentais da qualidade do RAG:
#   - faithfulness: a resposta está fundamentada nos contextos recuperados? (evita alucinação)
#   - answer_relevancy: a resposta é relevante para a pergunta feita?
#   - context_precision: os chunks recuperados são realmente úteis para responder?
#
# Para executar: python eval.py

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag.rag_service import RagService

# Perguntas de teste baseadas no conteúdo de dados.txt.
# Ajuste-as para refletir perguntas reais que os usuários fariam ao seu chatbot.
QUESTIONS = [
    "Quais são os principais tópicos abordados no documento?",
    "Qual é o objetivo principal descrito no conteúdo?",
    "Quais informações estão disponíveis sobre o tema central?",
]

GROUND_TRUTHS = [
    "O documento aborda os principais tópicos do contexto fornecido.",
    "O objetivo principal é fornecer informações sobre o tema central.",
    "O conteúdo contém informações detalhadas sobre o tema central.",
]


def run_evaluation():
    # Executa o pipeline completo de avaliação:
    # 1. Consulta o RAG para cada pergunta de teste
    # 2. Coleta resposta gerada e os chunks de contexto usados
    # 3. Monta um dataset no formato esperado pelo RAGAS
    # 4. Retorna um dicionário com os scores de cada métrica (valores entre 0 e 1)
    rag = RagService()

    questions, answers, contexts, ground_truths = [], [], [], []

    for question, truth in zip(QUESTIONS, GROUND_TRUTHS):
        response = rag.query(question)
        questions.append(question)
        answers.append(str(response))
        contexts.append([node.text for node in response.source_nodes])
        ground_truths.append(truth)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result
