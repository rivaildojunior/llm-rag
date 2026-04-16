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
    "Qual é o prazo para solicitar devolução por arrependimento na NovaShop?",
    "Quais são as condições para que uma devolução seja aceita?",
    "Como faço para cancelar um pedido que já foi enviado?",
    "Quais são as opções disponíveis para produto com defeito?",
    "Qual o prazo de reembolso para pagamentos via PIX ou boleto?",
    "Quais os canais de atendimento ao cliente e qual o horário?",
    "A NovaShop compartilha dados dos clientes com terceiros?",
    "Quando o frete é grátis?",
]

GROUND_TRUTHS = [
    "O cliente pode solicitar a devolução em até 7 dias corridos após o recebimento, conforme o Código de Defesa do Consumidor.",
    "O produto deve estar sem sinais de uso, na embalagem original e conter todos os acessórios e manuais.",
    "Se o pedido já foi enviado, o cliente deve aguardar a entrega e depois solicitar a devolução em até 7 dias corridos.",
    "O cliente pode escolher entre reembolso integral, troca por produto igual ou crédito para compras futuras.",
    "O reembolso via boleto ou PIX é feito por transferência bancária em até 10 dias úteis após a aprovação da devolução.",
    "Os canais são chat online, e-mail suporte@novashop.com e WhatsApp (11) 99999-9999, de segunda a sexta das 9h às 18h.",
    "Não. As informações dos clientes não são compartilhadas com terceiros e são usadas apenas para processamento de pedidos.",
    "O frete é grátis para compras acima de R$ 299,00 em regiões selecionadas.",
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
