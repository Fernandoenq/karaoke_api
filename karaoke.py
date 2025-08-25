# app/routers/karaoke.py
from fastapi import APIRouter
from app.services import analysis

router = APIRouter()

@router.get("/start")
def start_analysis(device: Optional[int | str] = None):
    """
    Inicia a captura e bloqueia até atingir os objetivos ou até o usuário interromper.
    """
    return analysis.start_analysis(device)

@router.get("/stop")
def stop_analysis():
    """
    Para a análise imediatamente e retorna o status.
    """
    return analysis.stop_analysis()

@router.get("/status")
def status():
    """
    Retorna o status atual da análise.
    """
    return analysis.status()
