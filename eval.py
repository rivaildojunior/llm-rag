from dotenv import load_dotenv
from rag.eval_service import run_evaluation

load_dotenv()

result = run_evaluation()
print(result)
