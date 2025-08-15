from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from .game_logic import GameEngine, SessionState, list_audio_devices
import asyncio
import json
import threading

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

# ====== HOME ======
@app.get("/")
def home():
    return {"ok": True, "service": "karaoke_api", "docs": "/docs"}

# ====== DEVICES ======
@app.get("/audio/devices")
def audio_devices():
    return list_audio_devices()

# ====== SSE: eventos ao vivo ======
@app.get("/live")
def live():
    """SSE com eventos ao vivo do engine (countdown, pitch, win, finished...)."""
    def event_gen():
        # dica de reconexão automática do navegador
        yield "retry: 500\n\n"
        while True:
            try:
                s = engine._live_q.get(timeout=1.0)  # string JSON
                yield f"data: {s}\n\n"
            except Exception:
                # keep-alive para não fechar a conexão
                yield "event: ping\ndata: {}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ====== START ======
@app.post("/start", response_model=StatusResponse, status_code=202)
async def start(
    bg: BackgroundTasks,
    music_path: Optional[str] = Query(None),
    in_dev: Optional[int] = Query(None),
    out_dev: Optional[int] = Query(None),
    wait_on_win: bool = Query(True),
):
    if engine.state not in (SessionState.idle, SessionState.finished):
        raise HTTPException(409, "Já existe uma sessão em andamento.")

    engine.prepare(music_path=music_path, in_dev=in_dev, out_dev=out_dev)

    # ❌ NÃO use BackgroundTasks aqui, pois você não retorna já.
    # bg.add_task(engine.run_session_async)

    # ✅ Inicie imediatamente em um thread
    threading.Thread(target=engine.run_session_async, daemon=True).start()

    if not wait_on_win:
        return StatusResponse(
            state=engine.state.value,
            countdown=engine.countdown_left(),
            progress=engine.progress(),
            message="Sessão iniciada"
        )

    # Aguarda vitória ou fim sem vitória (polling leve)
    timeout_s = 5 * 60
    tick = 0.1
    waited = 0.0
    while waited < timeout_s:
        if engine._win_event.is_set():
            from .main import ResultResponse  # ou mova para topo; apenas ilustrativo
            d = engine.result_summary()
            return ResultResponse(**d)  # 200

        if engine.state == SessionState.finished and not engine._win_event.is_set():
            return StatusResponse(
                state=engine.state.value,
                countdown=engine.countdown_left(),
                progress=engine.progress(),
                message="Sessão iniciada (sem vitória antecipada)"
            )

        await asyncio.sleep(tick)
        waited += tick

    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão iniciada (aguarda /result)"
    )
# ====== STOP ======
@app.post("/stop", response_model=StatusResponse)
def stop():
    engine.stop()
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão interrompida"
    )

# ====== STATUS ======
@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message=engine.status_message()
    )

# ====== RESULT ======
@app.get("/result", response_model=ResultResponse)
def result():
    if engine.state != SessionState.finished:
        raise HTTPException(409, "A sessão ainda não terminou.")
    return ResultResponse(**engine.result_summary())
