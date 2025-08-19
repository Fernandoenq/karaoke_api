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
import sounddevice as sd

from .game_logic import GameEngine, SessionState, list_audio_devices
from .start_ui import open_start_ui          # opcional
from .realtime_ui import launch_realtime_ui  # opcional


app = FastAPI(title="Karaokê API", version="1.0.0")

# CORS básico (libere domínios específicos em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GameEngine()
START_LOCK = asyncio.Lock()


# ---------------------- helpers ----------------------
def resolve_device_id(name_substr: str, want_input=True):
    """Resolve ID de dispositivo pelo nome (substring), priorizando WASAPI."""
    name_substr = name_substr.lower()
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()

    def hostapi_name(i):
        try:
            return hostapis[i]["name"].lower()
        except Exception:
            return ""

    cands = []
    for idx, d in enumerate(devs):
        has_ch = (d['max_input_channels'] if want_input else d['max_output_channels']) > 0
        if has_ch and name_substr in d['name'].lower():
            cands.append({
                "id": idx,
                "hostapi": hostapi_name(d["hostapi"]),
                "name": d["name"]
            })

    order = {"windows wasapi": 4, "windows wdm-ks": 3, "windows directsound": 2, "mme": 1}
    cands.sort(key=lambda c: order.get(c["hostapi"], 0), reverse=True)

    for c in cands:
        try:
            if want_input:
                sd.check_input_settings(device=c["id"], samplerate=44100, channels=1)
            else:
                sd.check_output_settings(device=c["id"], samplerate=44100, channels=2)
            return c["id"]
        except Exception:
            continue
    return None
# -----------------------------------------------------


# ====== MODELOS Pydantic ======
class StatusResponse(BaseModel):
    state: str
    countdown: Optional[int] = None
    progress: float = 0.0
    message: str = ""


class ResultResponse(BaseModel):
    state: str
    result: Optional[str] = None
    hit_ratio: Optional[float] = None
    max_streak_s: Optional[float] = None
    target_hz: Optional[float] = None
    tolerance_hz: Optional[float] = None
    achieved_pitch_hz: Optional[float] = None
    max_pitch_hz: Optional[float] = None
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    voiced_seconds: Optional[float] = None
    rms_mean: Optional[float] = None
    rms_median: Optional[float] = None
    jitter_cents: Optional[float] = None
    median_dev_cents: Optional[float] = None


StartResponse = Union[StatusResponse, ResultResponse]


# ====== HOME ======
@app.get("/")
def home():
    return {"ok": True, "service": "karaoke_api", "docs": "/docs"}


# ====== DEVICES ======
@app.get("/audio/devices")
def audio_devices():
    return list_audio_devices()


# ====== SSE: eventos ======
@app.get("/live")
def live():
    def event_gen():
        yield "retry: 500\n\n"
        while True:
            try:
                s = engine._live_q.get(timeout=1.0)
                yield f"data: {s}\n\n"
            except Exception:
                yield "event: ping\ndata: {}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ====== START ======
@app.post("/start", response_model=StartResponse, status_code=202)
async def start(
    bg: BackgroundTasks,
    response: Response,
    music_path: Optional[str] = Query(None),
    in_dev: Optional[int] = Query(None),
    out_dev: Optional[int] = Query(None),
    wait_on_win: bool = Query(True),
    in_name: Optional[str] = Query(None),
    out_name: Optional[str] = Query(None),
    with_ui: Optional[int] = Query(None),
):
    async with START_LOCK:
        # encerra sessão antiga
        if engine.state not in (SessionState.idle, SessionState.finished):
            engine.stop()
            for _ in range(30):
                if engine.state == SessionState.finished:
                    break
                await asyncio.sleep(0.1)
            if engine.state != SessionState.finished:
                engine.hard_reset()
                await asyncio.sleep(0.1)

        # resolve nomes se foi passado
        if in_name and in_dev is None:
            in_dev = resolve_device_id(in_name, want_input=True)
        if out_name and out_dev is None:
            out_dev = resolve_device_id(out_name, want_input=False)

        engine.prepare(music_path=music_path, in_dev=in_dev, out_dev=out_dev)

        if (with_ui == 1) or (os.environ.get("DEBUG_START_UI") == "1"):
            open_start_ui(engine)
            engine.publish(
                "state",
                state="countdown",
                in_dev=in_dev,
                out_dev=out_dev,
                music=music_path or "referencia_trecho.wav",
            )

        threading.Thread(target=engine.run_session_async, daemon=True).start()

        if not wait_on_win:
            return StatusResponse(
                state=engine.state.value,
                countdown=engine.countdown_left(),
                progress=engine.progress(),
                message="Sessão iniciada",
            )

        timeout_s = 5 * 60
        tick = 0.1
        waited = 0.0
        while waited < timeout_s:
            if engine._win_event.is_set():
                d = engine.result_summary()
                response.status_code = 200
                return ResultResponse(**d)

            if engine.state == SessionState.finished:
                d = engine.result_summary()
                if d.get("result") == "ganhou":
                    response.status_code = 200
                    return ResultResponse(**d)
                return StatusResponse(
                    state=engine.state.value,
                    countdown=engine.countdown_left(),
                    progress=engine.progress(),
                    message="Sessão finalizada (sem vitória)",
                )

            await asyncio.sleep(tick)
            waited += tick

        return StatusResponse(
            state=engine.state.value,
            countdown=engine.countdown_left(),
            progress=engine.progress(),
            message="Sessão iniciada (aguarda /result)",
        )


# ====== STOP ======
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


# ====== RESET ======
@app.post("/admin/reset")
def admin_reset():
    engine.hard_reset()
    return {"ok": True, "state": engine.state.value}


# ====== STATUS ======
@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        state=engine.state.value,
        countdown=engine.countdown_left(),
        progress=engine.progress(),
        message=engine.status_message(),
    )


# ====== RESULT ======
@app.get("/result", response_model=ResultResponse)
def result():
    if engine.state != SessionState.finished:
        raise HTTPException(409, "A sessão ainda não terminou.")
    return ResultResponse(**engine.result_summary())


# ====== TESTE ======
@app.post("/teste", response_model=StatusResponse, status_code=202)
def teste(
    in_dev: Optional[int] = Query(None),
    sr: int = Query(44100),
    blocksize: int = Query(2048),
):
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
