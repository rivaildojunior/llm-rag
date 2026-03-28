import re


class GuardrailsService:
    """Guardrails simples para validação de inputs e outputs."""
    
    # Palavras-chave suspeitas que indicam tentativa de injeção/jailbreak
    SUSPICIOUS_KEYWORDS = [
        "ignore", "esqueça", "forget", "bypass", "sistema", "admin",
        "root", "password", "senha", "delete", "drop", "execute"
    ]
    
    @staticmethod
    def validate_input(user_input: str) -> tuple[bool, str]:
        """
        Valida o input do usuário.
        Retorna: (é_válido, mensagem_de_erro)
        """
        if not user_input or len(user_input.strip()) == 0:
            return False, "❌ Input vazio. Digite algo."
        
        if len(user_input) > 2000:
            return False, "❌ Input muito longo (máx 2000 caracteres)."
        
        # Detecta tentativas simples de injeção
        if re.search(r'[<>"\';]', user_input[:50]):  # primeiros 50 chars
            return False, "❌ Caracteres suspeitos detectados."
        
        # Verifica palavras-chave suspeitas
        input_lower = user_input.lower()
        for keyword in GuardrailsService.SUSPICIOUS_KEYWORDS:
            if keyword in input_lower:
                return False, f"❌ Pergunta contém termo suspeito: '{keyword}'."
        
        return True, ""
    
    @staticmethod
    def validate_response(response, min_confidence: float = 0.3) -> tuple[bool, str]:
        """
        Valida se a resposta tem fontes confiáveis.
        Retorna: (é_válido, mensagem_de_alerta)
        """
        if not response.source_nodes:
            return False, "⚠️ Nenhuma fonte encontrada. Resposta pode não ser confiável."
        
        # Verifica score mínimo de relevância
        if hasattr(response.source_nodes[0], 'score'):
            top_score = response.source_nodes[0].score
            if top_score < min_confidence:
                return False, f"⚠️ Confiança baixa na resposta (score: {top_score:.2f})."
        
        return True, ""
