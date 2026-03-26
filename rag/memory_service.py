MAX_HISTORY_CHARS = 2000


class MemoryService:
    def __init__(self):
        # Histórico de conversa: lista de mensagens com role e conteúdo.
        self.chat_history = []

    def add(self, role, content):
        # Adiciona nova mensagem ao histórico (ex: role = 'user' ou 'assistant').
        self.chat_history.append({"role": role, "content": content})

    def get_context(self):
        # Retorna contexto concatenado com limite de caracteres.
        # Chamamos _trim_history para manter apenas a porção mais recente que cabe no limite.
        trimmed = self._trim_history(MAX_HISTORY_CHARS)
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in trimmed]
        )

    def _trim_history(self, max_chars):
        # Remove mensagens antigas até que o total de caracteres esteja abaixo do máximo.
        total = 0
        trimmed = []

        for msg in reversed(self.chat_history):
            msg_len = len(msg["content"])
            if total + msg_len > max_chars:
                break
            trimmed.append(msg)
            total += msg_len

        # Retorna em ordem cronológica (mais antiga primeiro).
        return list(reversed(trimmed))