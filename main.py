from dotenv import load_dotenv
from rag.chat_app import ChatApp
#teste
load_dotenv()

if __name__ == "__main__":
    app = ChatApp()
    app.run()