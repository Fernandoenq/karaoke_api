from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

# importa apenas o que existe no game_logic
from .game_logic import GameEngine, SessionState, list_audio_devices

app = FastAPI(title="Karaokê API", version="0.2.0")

# CORS (libera uso pelo seu front local; ajuste para produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # troque por ["http://localhost:3000"] quando tiver o front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== MODELOS Pydantic ======
class StatusResponse(BaseModel):
    state: SessionState
    countdown: Optional[int] = None
    progress: float = 0.0
    message: str = ""

class ResultResponse(BaseModel):
    state: SessionState
    result: Optional[str] = None  # "ganhou" | "perdeu"
    hit_ratio: Optional[float] = None
    max_streak_s: Optional[float] = None
    target_hz: Optional[float] = None
    tolerance_hz: Optional[float] = None
    achieved_pitch_hz: Optional[float] = None
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None

# ====== ENGINE (instância única) ======
engine = GameEngine()

# ====== ENDPOINTS ======

@app.get("/")
def home():
    return {"ok": True, "service": "karaoke_api", "docs": "/docs"}

@app.get("/audio/devices")
def audio_devices():
    """Lista dispositivos de áudio detectados pelo PortAudio (sounddevice)."""
    return list_audio_devices()

@app.get("/debug/audio-file")
def debug_audio_file(music_path: str):
    """
    Endpoint utilitário: confirma se o arquivo existe e mostra caminhos/size.
    Útil quando tiver dúvida de path relativo/absoluto.
    """
    p = Path(music_path).expanduser()
    return {
        "input": music_path,
        "exists": p.exists(),
        "resolved": str(p),
        "cwd": str(Path('.').resolve()),
        "size_bytes": (p.stat().st_size if p.exists() else None),
    }

@app.post("/start", response_model=StatusResponse, status_code=202)
async def start(
    bg: BackgroundTasks,
    music_path: Optional[str] = Query(None, description="Caminho do .wav a tocar"),
    in_dev: Optional[int] = Query(None, description="ID do dispositivo de entrada (microfone)"),
    out_dev: Optional[int] = Query(None, description="ID do dispositivo de saída (alto-falantes)"),
):
    """
    Inicia a sessão: faz countdown 3-2-1, toca o instrumental e mede o pitch.
    """
    if engine.state not in (SessionState.idle, SessionState.finished):
        raise HTTPException(409, "Já existe uma sessão em andamento")

    engine.prepare(music_path=music_path, in_dev=in_dev, out_dev=out_dev)

    # roda a sessão em background (countdown -> playing/medindo -> finished)
    bg.add_task(engine.run_session_async)

    return StatusResponse(
        state=engine.state,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão iniciada"
    )

@app.post("/stop", response_model=StatusResponse)
def stop():
    """Interrompe a sessão atual (se existir)."""
    engine.stop()
    return StatusResponse(
        state=engine.state,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão interrompida"
    )

@app.get("/status", response_model=StatusResponse)
def status():
    """Estado atual da sessão (idle | countdown | running | finished)."""
    return StatusResponse(
        state=engine.state,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message=engine.status_message()
    )

@app.get("/result", response_model=ResultResponse)
def result():
    """Resultado final — só disponível quando state == finished."""
    if engine.state != SessionState.finished:
        raise HTTPException(409, "A sessão ainda não terminou")
    return ResultResponse(**engine.result_summary())
