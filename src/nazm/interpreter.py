import logging
import nazm

logger = logging.getLogger(__name__)

class NazmInterpreter:
    def __init__(self):
        # Mapeamento Direto: Linguagem PT-BR -> Métodos do seu __init__.py
        self.commands = {
            "clicar": nazm.click,
            "esperar": nazm.wait_for,
            "clique_duplo": nazm.double_click,
            "clique_direito": nazm.right_click,
            "mover_para": nazm.move_to,
            "digitar": nazm.type_into,
            "existe": nazm.exists,
            "copiar": nazm.copy_from,
            "colar": nazm.paste_into,
            # Você pode adicionar aliases (apelidos) para a mesma função
            "aguardar": nazm.wait_for,
            "clique": nazm.click
        }

    def executar_linha(self, linha: str):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            return

        # Exemplo de entrada: clicar "botao_login"
        # partes[0] = "clicar", partes[1] = '"botao_login"'
        partes = linha.split(" ", 1)
        cmd_nome = partes[0].lower()
        argumento = partes[1].strip().strip('"') if len(partes) > 1 else None

        if cmd_nome in self.commands:
            func = self.commands[cmd_nome]
            try:
                if argumento:
                    # Se tiver argumento, chama a função (ex: nazm.click("botao_login"))
                    # O seu __init__ já vai cuidar de achar o path no AppData!
                    return func(argumento)
                else:
                    # Funções sem argumentos (ex: colar)
                    return func()
            except nazm.ElementNotFoundError as e:
                logger.error(f"Erro de Automação: Elemento '{argumento}' não apareceu.")
                raise
        else:
            logger.warning(f"Comando desconhecido: {cmd_nome}")