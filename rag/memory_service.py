MAX_HISTORY_CHARS = 2000


class MemoryService:
    def __init__(self):
        self.chat_history = []

    def add(self, role, content):
        self.chat_history.append({"role": role, "content": content})

    def get_context(self):
        trimmed = self._trim_history(MAX_HISTORY_CHARS)
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in trimmed]
        )

    def _trim_history(self, max_chars):
        total = 0
        trimmed = []

        for msg in reversed(self.chat_history):
            msg_len = len(msg["content"])
            if total + msg_len > max_chars:
                break
            trimmed.append(msg)
            total += msg_len

        return list(reversed(trimmed))