from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from .game_logic import GameEngine, SessionState, list_audio_devices

app = FastAPI(title="Karaokê API", version="1.0.0")

# CORS básico (libere domínios específicos em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para o(s) host(s) do seu front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GameEngine()

# ====== MODELOS Pydantic (somente tipos nativos) ======
class StatusResponse(BaseModel):
    state: str
    countdown: Optional[int] = None
    progress: float = 0.0
    message: str = ""

class ResultResponse(BaseModel):
    state: str
    result: Optional[str] = None          # "ganhou" | "perdeu"
    hit_ratio: Optional[float] = None
    max_streak_s: Optional[float] = None
    target_hz: Optional[float] = None
    tolerance_hz: Optional[float] = None
    achieved_pitch_hz: Optional[float] = None
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None

@app.get("/")
def home():
    return {"ok": True, "service": "karaoke_api", "docs": "/docs"}

@app.get("/audio/devices")
def audio_devices():
    return list_audio_devices()

@app.post("/start", response_model=StatusResponse, status_code=202)
async def start(
    bg: BackgroundTasks,
    music_path: Optional[str] = Query(None, description="Caminho absoluto ou relativo para o .wav"),
    in_dev: Optional[int] = Query(None, description="ID do dispositivo de entrada (microfone)"),
    out_dev: Optional[int] = Query(None, description="ID do dispositivo de saída (alto‑falantes)")
):
    if engine.state not in (SessionState.idle, SessionState.finished):
        raise HTTPException(409, "Já existe uma sessão em andamento.")

    engine.prepare(music_path=music_path, in_dev=in_dev, out_dev=out_dev)
    bg.add_task(engine.run_session_async)
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão iniciada"
    )

@app.post("/stop", response_model=StatusResponse)
def stop():
    engine.stop()
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão interrompida"
    )

@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message=engine.status_message()
    )

@app.get("/result", response_model=ResultResponse)
def result():
    if engine.state != SessionState.finished:
        raise HTTPException(409, "A sessão ainda não terminou.")
    return ResultResponse(**engine.result_summary())
