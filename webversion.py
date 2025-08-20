# app.py
from __future__ import annotations

import os, time, json, math, wave, threading
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np
import speech_recognition as sr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from uuid import uuid4

from fastapi.middleware.cors import CORSMiddleware

# ===================== ÁUDIO/PITCH - parâmetros =====================
recognizer = sr.Recognizer()

DEFAULT_TARGET_PITCH_HZ = 196.0     # G3
DEFAULT_TOLERANCE_CENTS = 300.0     # ±3 semitons
DEFAULT_MIN_VOLUME_DBFS = -55.0     # aceita falar baixo
SMOOTHING_FRAMES = 1
FMIN, FMAX = 60.0, 1200.0
VAD_FACTOR = 1.1
FRAME_SECONDS = 0.25
EPS = 1e-12

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ===================== Helpers =====================
def frame_from_audio(audio_data: sr.AudioData) -> np.ndarray:
    x = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32)
    return x / 32768.0

def rms_dbfs(x: np.ndarray) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(x**2) + EPS))
    db = 20.0 * np.log10(rms + EPS)
    return rms, db

def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(max(f, 1e-6) / 440.0)

def nearest_note_name(f: float) -> Tuple[str, int]:
    m = round(hz_to_midi(f))
    name = NOTE_NAMES[int(m) % 12]
    octave = int(m // 12) - 1
    return name, octave

def cents_error(f_est: float, f_ref: float) -> float:
    return 1200.0 * math.log2(max(f_est, 1e-6) / max(f_ref, 1e-6))

def estimate_pitch_fft(x: np.ndarray, sr_hz: int, fmin=FMIN, fmax=FMAX) -> Optional[float]:
    x = x - np.mean(x)
    if np.max(np.abs(x)) < 1e-6:
        return None
    w = np.hanning(len(x))
    xw = x * w
    n = int(2 ** np.ceil(np.log2(len(xw))))
    spec = np.fft.rfft(xw, n=n)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr_hz)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(idx) < 3:
        return None
    sub_mag = mag[idx]
    k_local = int(np.argmax(sub_mag))
    k = int(idx[k_local])
    if 1 <= k < len(mag) - 1:
        a = mag[k - 1]; b = mag[k]; c = mag[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        refined_bin = k + delta
        f0 = refined_bin * (sr_hz / n)
    else:
        f0 = freqs[k]
    return float(f0)

def get_audio_duration_seconds(path: str) -> float:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return float(frames) / float(sr)
    except Exception as e:
        raise RuntimeError(
            "Não foi possível obter a duração do áudio. "
            "Tente instalar 'soundfile' (pip install soundfile). "
            f"Erro: {e}"
        )

# ===================== Estado de sessão (única) =====================
class SessionResult(BaseModel):
    result: str
    elapsed_seconds: float
    hit: bool
    pitch_hz: Optional[float] = None
    error_cents: Optional[float] = None
    volume_dbfs: Optional[float] = None
    nearest_note: Optional[str] = None
    target_pitch_hz: float
    tolerance_cents: float
    min_volume_dbfs: float
    music_path: str

class SessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.stop_flag = threading.Event()
        self.started_epoch = 0.0
        self.end_epoch = 0.0
        self.sid = ""
        self.cfg = {
            "music_path": "",
            "listen_seconds": 0.0,
            "target_pitch_hz": DEFAULT_TARGET_PITCH_HZ,
            "tolerance_cents": DEFAULT_TOLERANCE_CENTS,
            "min_volume_dbfs": DEFAULT_MIN_VOLUME_DBFS,
        }
        self.result: Optional[SessionResult] = None

    def reset(self):
        self.stop_flag = threading.Event()
        self.started_epoch = 0.0
        self.end_epoch = 0.0
        self.result = None
        self.sid = ""

SESSION = SessionState()

# ===================== Worker de captura (sem UI) =====================
def capture_worker():
    cfg = SESSION.cfg
    listen_seconds = cfg["listen_seconds"]
    target_pitch = cfg["target_pitch_hz"]
    tolerance = cfg["tolerance_cents"]
    min_vol = cfg["min_volume_dbfs"]

    pitch_buf: Deque[float] = deque(maxlen=SMOOTHING_FRAMES)
    best = {
        "best_pitch_hz": None,
        "best_err_abs": None,
        "best_dbfs": None,
        "nearest_note": None,
    }

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            noise_audio = recognizer.listen(source, phrase_time_limit=1.0)
            noise = frame_from_audio(noise_audio)
            noise_rms, _noise_dbfs = rms_dbfs(noise)

            start_t = time.time()
            SESSION.started_epoch = start_t

            while not SESSION.stop_flag.is_set():
                elapsed = time.time() - start_t
                if elapsed >= listen_seconds:
                    break

                audio = recognizer.listen(source, phrase_time_limit=FRAME_SECONDS)
                x = frame_from_audio(audio)
                sr_hz = source.SAMPLE_RATE

                rms, dbfs = rms_dbfs(x)
                speaking = rms > (noise_rms * VAD_FACTOR)

                f0 = None
                if speaking:
                    f0 = estimate_pitch_fft(x, sr_hz)
                    if f0 is not None:
                        pitch_buf.append(f0)

                f0_smooth = float(np.median(pitch_buf)) if len(pitch_buf) else None

                err_cents = None
                err_abs = None
                near = None
                if speaking and f0_smooth is not None:
                    err_cents = cents_error(f0_smooth, target_pitch)
                    err_abs = abs(err_cents)
                    name, octv = nearest_note_name(f0_smooth)
                    near = f"{name}{octv}"

                # vitória (critérios simples)
                if (f0_smooth is not None) and (err_abs is not None):
                    if (dbfs >= min_vol) and (err_abs <= tolerance):
                        SESSION.result = SessionResult(
                            result="ganhou",
                            elapsed_seconds=round(time.time() - start_t, 3),
                            hit=True,
                            pitch_hz=f0_smooth,
                            error_cents=err_cents,
                            volume_dbfs=dbfs,
                            nearest_note=near,
                            target_pitch_hz=target_pitch,
                            tolerance_cents=tolerance,
                            min_volume_dbfs=min_vol,
                            music_path=cfg["music_path"],
                        )
                        break

        SESSION.end_epoch = time.time()
        if SESSION.result is None:
            SESSION.result = SessionResult(
                result="nao-ganhou",
                elapsed_seconds=round(SESSION.end_epoch - SESSION.started_epoch, 3),
                hit=False,
                pitch_hz=best["best_pitch_hz"],
                error_cents=None,
                volume_dbfs=best["best_dbfs"] if best["best_dbfs"] is not None else None,
                nearest_note=best["nearest_note"],
                target_pitch_hz=target_pitch,
                tolerance_cents=tolerance,
                min_volume_dbfs=min_vol,
                music_path=cfg["music_path"],
            )

    except Exception:
        SESSION.end_epoch = time.time()
        SESSION.result = SessionResult(
            result="aborted",
            elapsed_seconds=round(SESSION.end_epoch - SESSION.started_epoch, 3) if SESSION.started_epoch else 0.0,
            hit=False,
            pitch_hz=None,
            error_cents=None,
            volume_dbfs=None,
            nearest_note=None,
            target_pitch_hz=DEFAULT_TARGET_PITCH_HZ,
            tolerance_cents=DEFAULT_TOLERANCE_CENTS,
            min_volume_dbfs=DEFAULT_MIN_VOLUME_DBFS,
            music_path=SESSION.cfg.get("music_path", ""),
        )
    finally:
        with SESSION.lock:
            SESSION.is_running = False

# ===================== FastAPI =====================
app = FastAPI(title="Karaokê Judge API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.get("/health")
def health():
    return {"ok": True}

# -------- Sessão tradicional (sem UI) --------
@app.post("/start")
def start(request: Request):
    params = request.query_params
    music_path = params.get("music_path")
    if not music_path:
        raise HTTPException(status_code=400, detail="Parâmetro 'music_path' é obrigatório.")

    try:
        listen_seconds = float(get_audio_duration_seconds(music_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with SESSION.lock:
        if SESSION.is_running:
            return JSONResponse(status_code=409, content={"error": "already-running"})

        SESSION.reset()
        SESSION.sid = str(uuid4())
        SESSION.cfg.update(
            music_path=music_path,
            listen_seconds=listen_seconds,
            target_pitch_hz=DEFAULT_TARGET_PITCH_HZ,
            tolerance_cents=DEFAULT_TOLERANCE_CENTS,
            min_volume_dbfs=DEFAULT_MIN_VOLUME_DBFS,
        )
        SESSION.is_running = True

        th = threading.Thread(target=capture_worker, daemon=True)
        th.start()

    return JSONResponse(status_code=202, content={"state": "started", "sid": SESSION.sid})

@app.post("/stop")
def stop(request: Request):
    force = request.query_params.get("force", "false").lower() == "true"
    with SESSION.lock:
        if not SESSION.is_running and not force:
            return {"state": "idle"}
        SESSION.stop_flag.set()
    return {"state": "stopping" if SESSION.is_running else "idle"}

@app.get("/result")
def result():
    with SESSION.lock:
        if SESSION.is_running:
            return JSONResponse(status_code=202, content={"state": "running"})
        if SESSION.result is None:
            raise HTTPException(status_code=404, detail="Sem resultado disponível.")
        return SESSION.result.model_dump()

# -------- Utilitário simples (sem UI): mede alguns segundos e retorna JSON --------
@app.get("/teste")
def teste(request: Request):
    params = request.query_params
    duration_seconds = _parse_float(params.get("duration_seconds", 5.0), 5.0)
    frame_seconds    = _parse_float(params.get("frame_seconds", FRAME_SECONDS), FRAME_SECONDS)
    with_pitch = params.get("with_pitch", "true").lower() in {"1", "true", "yes", "y"}

    if duration_seconds <= 0 or frame_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds e frame_seconds devem ser > 0.")
    ...


    with_pitch = params.get("with_pitch", "true").lower() in {"1", "true", "yes", "y"}

    if duration_seconds <= 0 or frame_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds e frame_seconds devem ser > 0.")

    samples = []
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)

            start_t = time.time()
            noise_audio = recognizer.listen(source, phrase_time_limit=min(0.8, frame_seconds))
            noise = frame_from_audio(noise_audio)
            noise_rms, noise_dbfs = rms_dbfs(noise)

            while (time.time() - start_t) < duration_seconds:
                audio = recognizer.listen(source, phrase_time_limit=frame_seconds)
                x = frame_from_audio(audio)
                sr_hz = source.SAMPLE_RATE

                rms, dbfs = rms_dbfs(x)
                speaking = rms > (noise_rms * VAD_FACTOR)

                f0 = None
                near = None
                if with_pitch and speaking:
                    f0 = estimate_pitch_fft(x, sr_hz)
                    if f0 is not None and f0 > 0:
                        name, octv = nearest_note_name(f0)
                        near = f"{name}{octv}"

                samples.append({
                    "t_rel": round(time.time() - start_t, 3),
                    "dbfs": dbfs,
                    "speaking": bool(speaking),
                    "vad_thresh_dbfs": noise_dbfs,
                    "pitch_hz": float(f0) if f0 else None,
                    "nearest_note": near,
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler o microfone: {e}")

    return {
        "ok": True,
        "duration_seconds": duration_seconds,
        "frame_seconds": frame_seconds,
        "with_pitch": with_pitch,
        "samples": samples,
    }

# -------- NOVO: Stream SSE independente para UI Web --------
def _parse_float(v, default):
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return default

@app.get("/meter/live")
def meter_live(request: Request):
    params = request.query_params
    duration_seconds = _parse_float(params.get("duration_seconds", 20.0), 20.0)
    frame_seconds   = _parse_float(params.get("frame_seconds", FRAME_SECONDS), FRAME_SECONDS)
    fmin            = _parse_float(params.get("fmin", FMIN), FMIN)
    fmax            = _parse_float(params.get("fmax", FMAX), FMAX)
    with_pitch = params.get("with_pitch", "true").lower() in {"1", "true", "yes", "y"}

    if duration_seconds <= 0 or frame_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds e frame_seconds devem ser > 0.")
    ...


    def event_stream():
        yield "retry: 800\n\n"
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                start_t = time.time()

                noise_audio = recognizer.listen(source, phrase_time_limit=min(0.8, frame_seconds))
                noise = frame_from_audio(noise_audio)
                noise_rms, noise_dbfs = rms_dbfs(noise)

                while (time.time() - start_t) < duration_seconds:
                    audio = recognizer.listen(source, phrase_time_limit=frame_seconds)
                    x = frame_from_audio(audio)
                    sr_hz = source.SAMPLE_RATE

                    rms, dbfs = rms_dbfs(x)
                    speaking = rms > (noise_rms * VAD_FACTOR)

                    f0 = None
                    near = None
                    err_cents = None

                    if with_pitch and speaking:
                        f0 = estimate_pitch_fft(x, sr_hz, fmin=fmin, fmax=fmax)
                        if f0 is not None and f0 > 0:
                            name, octv = nearest_note_name(f0)
                            near = f"{name}{octv}"
                            err_cents = cents_error(f0, DEFAULT_TARGET_PITCH_HZ)

                    payload = {
                        "t_rel": round(time.time() - start_t, 3),
                        "dbfs": dbfs,
                        "speaking": bool(speaking),
                        "vad_thresh_dbfs": noise_dbfs,
                        "pitch_hz": float(f0) if f0 else None,
                        "nearest_note": near,
                        "error_cents": err_cents,
                        "target_pitch_hz": DEFAULT_TARGET_PITCH_HZ,
                        "tolerance_cents": DEFAULT_TOLERANCE_CENTS,
                        "min_volume_dbfs": DEFAULT_MIN_VOLUME_DBFS,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            err = {"error": f"Falha ao ler o microfone: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), headers=headers)

# -------- NOVO: Página Web do monitor --------
@app.get("/panel")
def panel():
    """
    Página HTML que consome /meter/live e exibe medidas em tempo real.
    """
    html = f"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <title>Karaokê • Monitor Web</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#0b1020; color:#e5e7eb; }}
    .wrap {{ max-width: 760px; margin: 32px auto; padding: 16px; }}
    .card {{ background:#111827; border:1px solid #1f2937; border-radius:16px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.25); }}
    .row {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .k {{ color:#94a3b8; font-size:13px; }}
    .v {{ font-size:28px; font-weight:700; }}
    .controls {{ display:flex; gap:8px; align-items:center; margin-bottom:16px; flex-wrap:wrap; }}
    input, select, button {{ background:#0b1220; color:#e5e7eb; border:1px solid #1f2937; border-radius:10px; padding:8px 10px; }}
    button {{ cursor:pointer; }}
    canvas {{ width:100%; height:140px; background:#0b1220; border:1px solid #1f2937; border-radius:10px; }}
    small {{ color:#94a3b8; }}
    a {{ color:#93c5fd; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="controls">
        <label>Duration (s) <input type="number" id="dur" value="20" min="1" step="1"></label>
        <label>Frame (s) <input type="number" id="frame" value="{FRAME_SECONDS}" min="0.05" step="0.05"></label>
        <label>Pitch?
          <select id="pitch">
            <option value="true" selected>sim</option>
            <option value="false">não</option>
          </select>
        </label>
        <button id="startBtn">Start</button>
        <button id="stopBtn">Stop</button>
        <small id="status">parado</small>
      </div>

      <div class="row">
        <div>
          <div class="k">Volume (dBFS)</div>
          <div class="v" id="dbfs">–</div>
        </div>
        <div>
          <div class="k">Pitch (Hz)</div>
          <div class="v" id="pitch_hz">–</div>
        </div>
        <div>
          <div class="k">Nota</div>
          <div class="v" id="note">–</div>
        </div>
      </div>

      <div style="margin-top:20px;">
        <div class="k">Sinal (últimos 15 s)</div>
        <canvas id="chart" width="720" height="160"></canvas>
        <div class="k" style="margin-top:8px;">Dica: use um único processo do servidor (sem <code>--workers</code>), e não abra múltiplas abas consumindo o microfone ao mesmo tempo.</div>
      </div>
    </div>
  </div>

<script>
let es = null;
let points = [];
const maxSeconds = 15;

function drawChart() {{
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if (points.length < 2) return;

  const tMin = Math.max(0, points[points.length-1][0] - maxSeconds);
  const tMax = points[points.length-1][0];
  const x = v => ( (v - tMin) / (tMax - tMin) ) * (c.width - 20) + 10;

  // dBFS normalizado (-80..0) -> 0..1
  const dbMin = -80, dbMax = 0;
  const y = v => {{
    const nv = (v - dbMin) / (dbMax - dbMin);
    return (1 - Math.max(0, Math.min(1, nv))) * (c.height - 20) + 10;
  }};

  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (const [t, db] of points) {{
    if (t < tMin) continue;
    const px = x(t), py = y(db);
    if (!started) {{ ctx.moveTo(px, py); started = true; }}
    else ctx.lineTo(px, py);
  }}
  ctx.strokeStyle = '#60a5fa';
  ctx.stroke();

  // Eixos
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(10, 10); ctx.lineTo(10, c.height-10); ctx.lineTo(c.width-10, c.height-10);
  ctx.stroke();
}}

function stopStream() {{
  if (es) {{
    es.close();
    es = null;
  }}
  document.getElementById('status').textContent = 'parado';
}}

document.getElementById('startBtn').onclick = () => {{
  stopStream();
  points = [];

  const dur = parseFloat(document.getElementById('dur').value || '20');
  const frame = parseFloat(document.getElementById('frame').value || '{FRAME_SECONDS}');
  const withPitch = document.getElementById('pitch').value;

  const url = `/meter/live?duration_seconds=${{dur}}&frame_seconds=${{frame}}&with_pitch=${{withPitch}}`;
  es = new EventSource(url);
  document.getElementById('status').textContent = 'capturando...';

  es.onmessage = (e) => {{
    try {{
      const data = JSON.parse(e.data);
      if (data.error) {{
        document.getElementById('status').textContent = 'erro: ' + data.error;
        stopStream();
        return;
      }}
      if (typeof data.dbfs === 'number') {{
        document.getElementById('dbfs').textContent = data.dbfs.toFixed(1);
        const t = typeof data.t_rel === 'number' ? data.t_rel : (points.length? points[points.length-1][0] + frame: 0);
        points.push([t, data.dbfs]);
        const cut = Math.max(0, points.length - Math.ceil(maxSeconds / frame) - 5);
        if (cut > 0) points.splice(0, cut);
        drawChart();
      }}
      document.getElementById('pitch_hz').textContent = data.pitch_hz ? data.pitch_hz.toFixed(1) : '–';
      document.getElementById('note').textContent = data.nearest_note || '–';
    }} catch (err) {{
      console.error(err);
    }}
  }};

  es.onerror = () => {{
    document.getElementById('status').textContent = 'conexão perdida';
    stopStream();
  }};
}};

document.getElementById('stopBtn').onclick = stopStream;
</script>
</body>
</html>
    """
    return HTMLResponse(content=html)
