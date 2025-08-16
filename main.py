# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Union
import asyncio
import json
import threading
import os

from .game_logic import GameEngine, SessionState, list_audio_devices
from .start_ui import open_start_ui          # opcional: janela de sessão (/start)
from .realtime_ui import launch_realtime_ui  # opcional: janela de análise em tempo real (/teste)



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

# Lock para serializar /start e evitar corridas
START_LOCK = asyncio.Lock()

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
    max_pitch_hz: Optional[float] = None  # 👈 NOVO: maior f0 válido alcançado
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    # métricas extras que o engine pode anexar:
    voiced_seconds: Optional[float] = None
    rms_mean: Optional[float] = None
    rms_median: Optional[float] = None
    jitter_cents: Optional[float] = None
    median_dev_cents: Optional[float] = None

# Para o /start retornar 202 (status) ou 200 (vitória) com payloads diferentes
StartResponse = Union[StatusResponse, ResultResponse]

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
# ====== START ======
@app.post("/start", response_model=StartResponse, status_code=202)
async def start(
    bg: BackgroundTasks,
    response: Response,
    music_path: Optional[str] = Query(None),
    in_dev: Optional[int] = Query(None),
    out_dev: Optional[int] = Query(None),
    wait_on_win: bool = Query(True),
    # 👇 opcional (não quebra contrato): só liga UI se você pedir
    with_ui: Optional[int] = Query(None, description="1 para abrir UI local"),
):
    async with START_LOCK:
        # --- Pré-empt: encerra qualquer sessão antiga antes de iniciar outra ---
        if engine.state not in (SessionState.idle, SessionState.finished):
            engine.stop()  # tentativa graciosa
            # aguarda até 3s pelo término (em steps de 100ms)
            for _ in range(30):
                if engine.state == SessionState.finished:
                    break
                await asyncio.sleep(0.1)
            # se ainda não terminou, força reset e dá folga p/ soltar dispositivos
            if engine.state != SessionState.finished:
                engine.hard_reset()
                await asyncio.sleep(0.1)

        # --- Prepara e inicia nova sessão ---
        engine.prepare(music_path=music_path, in_dev=in_dev, out_dev=out_dev)

        # se pediu UI, registra e semeia estado atual (abre só uma vez)
        if (with_ui == 1) or (os.environ.get("DEBUG_START_UI") == "1"):
            open_start_ui(engine)
            engine.publish(
                "state",
                state="countdown",
                in_dev=in_dev,
                out_dev=out_dev,
                music=music_path or "referencia_trecho.wav",
            )

        # inicia a thread da sessão
        threading.Thread(target=engine.run_session_async, daemon=True).start()

        # retorno imediato se não quiser aguardar vitória
        if not wait_on_win:
            return StatusResponse(
                state=engine.state.value,
                countdown=engine.countdown_left(),
                progress=engine.progress(),
                message="Sessão iniciada",
            )

        # --- Aguarda vitória antecipada ou término ---
        timeout_s = 5 * 60
        tick = 0.1
        waited = 0.0
        while waited < timeout_s:
            # vitória antecipada durante a sessão
            if engine._win_event.is_set():
                d = engine.result_summary()
                response.status_code = 200
                return ResultResponse(**d)

            # sessão terminou: avalia resultado final
            if engine.state == SessionState.finished:
                d = engine.result_summary()
                if d.get("result") == "ganhou":
                    response.status_code = 200  # vitória ao finalizar
                    return ResultResponse(**d)
                # terminou sem vitória → mantém 202
                return StatusResponse(
                    state=engine.state.value,
                    countdown=engine.countdown_left(),
                    progress=engine.progress(),
                    message="Sessão finalizada (sem vitória antecipada)",
                )

            await asyncio.sleep(tick)
            waited += tick

        # timeout: sessão pode seguir rodando; front deve consultar /result
        return StatusResponse(
            state=engine.state.value,
            countdown=engine.countdown_left(),
            progress=engine.progress(),
            message="Sessão iniciada (aguarda /result)",
        )


# STOP com modo força
@app.post("/stop", response_model=StatusResponse)
def stop(force: bool = Query(False)):
    if force:
        engine.hard_reset()
        return StatusResponse(
            state=engine.state.value,
            countdown=engine.countdown_left(),
            progress=engine.progress(),
            message="Sessão abortada (force)"
        )
    engine.stop()
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message="Sessão interrompida"
    )

# RESET administrativo (sempre força idle)
@app.post("/admin/reset")
def admin_reset():
    engine.hard_reset()
    return {"ok": True, "state": engine.state.value}

# ====== STATUS ======
@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countown_left() if hasattr(engine, "countown_left") else engine.countdown_left(),
        progress=engine.progress(),
        message=engine.status_message(),
    )

# ====== RESULT ======
@app.get("/result", response_model=ResultResponse)
def result():
    if engine.state != SessionState.finished:
        raise HTTPException(409, "A sessão ainda não terminou.")
    return ResultResponse(**engine.result_summary())

# ====== TESTE (abre UI de análise em tempo real) ======
@app.post("/teste", response_model=StatusResponse, status_code=202)
def teste(
    in_dev: Optional[int] = Query(None, description="ID do dispositivo de entrada (veja /audio/devices)"),
    sr: int = Query(44100, description="Sample rate"),
    blocksize: int = Query(2048, description="Tamanho do bloco em amostras"),
):
    """
    Abre uma janela simples (Tkinter) mostrando a análise em tempo real do microfone.
    Fecha a janela => encerra a captura. A API continua disponível.
    """
    # dispara a UI em thread separada
    threading.Thread(
        target=launch_realtime_ui,
        args=(in_dev, sr, blocksize),
        daemon=True
    ).start()

    return StatusResponse(
        state="running",
        countdown=None,
        progress=0.0,
        message="UI de análise em tempo real iniciada",
    )
