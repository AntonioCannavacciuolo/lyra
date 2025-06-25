# Lyra/Agents/SunaWrapper.py

from suna.agent import SunaAgent
from suna.toolbox.default import default_tools

# Inizializza SUNA
agent = SunaAgent(tools=default_tools())

def execute_task(task_prompt: str) -> str:
    """
    Esegue un task usando SUNA e restituisce l'output.
    """
    try:
        result = agent.run(task_prompt)
        return result.output
    except Exception as e:
        return f"[Errore SUNA]: {e}"
