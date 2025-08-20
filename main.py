# app.py
from __future__ import annotations

import os
import time
import json
import math
import wave
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np
import speech_recognition as sr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===================== RECONHECIMENTO / ÁUDIO =====================
recognizer = sr.Recognizer()

# ===================== PARÂMETROS DO JOGO (default) =====================
DEFAULT_TARGET_PITCH_HZ = 330.0    # E4 (Mi4) ~ exemplo
DEFAULT_TOLERANCE_CENTS = 35.0     # tolerância de acerto
DEFAULT_MIN_VOLUME_DBFS = -35.0    # volume mínimo
SMOOTHING_FRAMES = 5               # suavização de pitch (mediana)
FMIN, FMAX = 70.0, 1000.0          # faixa de pitch
VAD_FACTOR = 1.5                   # fala se RMS > (ruído * VAD_FACTOR)
FRAME_SECONDS = 0.35               # “latência” por bloco
EPS = 1e-12

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ===================== MODELOS Pydantic =====================
class StartRequest(BaseModel):
    music_path: str = Field(..., description="Caminho do arquivo de áudio da música (usado para saber a duração total de escuta).")
    resultado_path: str = Field(..., description="Caminho de arquivo onde o back vai salvar o melhor resultado observado (JSON).")
    target_pitch_hz: float = Field(DEFAULT_TARGET_PITCH_HZ, description="Nota-alvo em Hz.")
    tolerance_cents: float = Field(DEFAULT_TOLERANCE_CENTS, description="Tolerância em cents para considerar acerto (HIT).")
    min_volume_dbfs: float = Field(DEFAULT_MIN_VOLUME_DBFS, description="Volume mínimo (dBFS) para validar o acerto.")
    device_index: Optional[int] = Field(None, description="Índice do microfone (opcional).")
    # Se quiser forçar uma duração e ignorar a música:
    override_listen_seconds: Optional[float] = Field(None, description="Se fornecido, usa essa duração em segundos em vez da duração da música.")

class StartResponse(BaseModel):
    status: str
    message: str
    elapsed_seconds: float
    hit: bool
    # métrica final observada (suavizada) no momento do hit ou ao final
    pitch_hz: Optional[float] = None
    error_cents: Optional[float] = None
    volume_dbfs: Optional[float] = None
    nearest_note: Optional[str] = None
    resultado_path: str

# ===================== UTILITÁRIOS DE ÁUDIO =====================
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
    k_local = np.argmax(sub_mag)
    k = idx[k_local]

    if 1 <= k < len(mag) - 1:
        a = mag[k - 1]
        b = mag[k]
        c = mag[k + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        refined_bin = k + delta
        f0 = refined_bin * (sr_hz / n)
    else:
        f0 = freqs[k]

    return float(f0)

def get_audio_duration_seconds(path: str) -> float:
    """
    Retorna a duração do arquivo de música em segundos.
    - Tenta 'soundfile' (se instalado) para múltiplos formatos.
    - Se não tiver, tenta WAV nativo via 'wave'.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Tenta via soundfile (mais formatos)
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    # Fallback: WAV
    try:
        with wave.open(path, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return float(frames) / float(sr)
    except Exception as e:
        raise RuntimeError(
            f"Não foi possível obter a duração do áudio. "
            f"Tente instalar 'soundfile' (pip install soundfile). Erro: {e}"
        )

def ensure_dir_for_file(file_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

def save_result_json(resultado_path: str, payload: dict) -> None:
    ensure_dir_for_file(resultado_path)
    with open(resultado_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ===================== FASTAPI APP =====================
app = FastAPI(title="Karaokê Judge API", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start", response_model=StartResponse)
def start(req: StartRequest):
    """
    Bloqueia até:
      - Acertar (HIT) => retorna imediatamente com status='hit'
      - Ou acabar o tempo (duração da música / override) => salva resultado e retorna 'no-hit-saved'
    Também grava JSON no `resultado_path` com o melhor desempenho observado.
    """
    # validações iniciais
    try:
        listen_seconds = (
            float(req.override_listen_seconds)
            if req.override_listen_seconds is not None
            else float(get_audio_duration_seconds(req.music_path))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if listen_seconds <= 0:
        raise HTTPException(status_code=400, detail="Duração inválida (<= 0).")

    # buffers/estado de melhor desempenho
    pitch_buf: Deque[float] = deque(maxlen=SMOOTHING_FRAMES)
    best = {
        "best_pitch_hz": None,          # maior proximidade da alvo? aqui vamos salvar o pitch suavizado mais alto alcançado
        "best_error_cents_abs": None,   # menor |erro| em relação ao alvo
        "best_volume_dbfs": None,
        "nearest_note": None,
        "hit": False,
    }

    # loop de captura
    start_t = time.time()

    # Seleção de microfone
    mic_kwargs = {}
    if req.device_index is not None:
        mic_kwargs["device_index"] = req.device_index

    try:
        with sr.Microphone(**mic_kwargs) as source:
            # calibração de ruído breve
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            noise_audio = recognizer.listen(source, phrase_time_limit=1.0)
            noise = frame_from_audio(noise_audio)
            noise_rms, _noise_dbfs = rms_dbfs(noise)

            while True:
                elapsed = time.time() - start_t
                if elapsed >= listen_seconds:
                    break

                audio = recognizer.listen(source, phrase_time_limit=FRAME_SECONDS)
                x = frame_from_audio(audio)
                sr_hz = source.SAMPLE_RATE

                # loudness + VAD
                rms, dbfs = rms_dbfs(x)
                speaking = rms > (noise_rms * VAD_FACTOR)

                # pitch
                f0 = estimate_pitch_fft(x, sr_hz)
                if f0 is not None and speaking:
                    pitch_buf.append(f0)

                f0_smooth = float(np.median(pitch_buf)) if len(pitch_buf) > 0 else None

                if speaking and f0_smooth is not None:
                    err_cents = cents_error(f0_smooth, req.target_pitch_hz)
                    err_abs = abs(err_cents)
                    near_name, near_oct = nearest_note_name(f0_smooth)
                    near_str = f"{near_name}{near_oct}"

                    # atualizar melhor desempenho
                    if (best["best_error_cents_abs"] is None) or (err_abs < best["best_error_cents_abs"]):
                        best["best_error_cents_abs"] = err_abs
                        best["best_pitch_hz"] = f0_smooth
                        best["best_volume_dbfs"] = dbfs
                        best["nearest_note"] = near_str

                    # checa acerto
                    if (dbfs >= req.min_volume_dbfs) and (err_abs <= req.tolerance_cents):
                        # registramos resultado e retornamos HIT
                        payload = {
                            "status": "hit",
                            "elapsed_seconds": round(time.time() - start_t, 3),
                            "hit": True,
                            "pitch_hz": f0_smooth,
                            "error_cents": err_cents,
                            "volume_dbfs": dbfs,
                            "nearest_note": near_str,
                            "target_pitch_hz": req.target_pitch_hz,
                            "tolerance_cents": req.tolerance_cents,
                            "min_volume_dbfs": req.min_volume_dbfs,
                            "music_path": req.music_path,
                        }
                        save_result_json(req.resultado_path, payload)
                        return StartResponse(
                            status="hit",
                            message="Nota atingida dentro da tolerância e volume mínimo.",
                            elapsed_seconds=payload["elapsed_seconds"],
                            hit=True,
                            pitch_hz=f0_smooth,
                            error_cents=err_cents,
                            volume_dbfs=dbfs,
                            nearest_note=near_str,
                            resultado_path=req.resultado_path,
                        )

        # se chegou aqui, não houve HIT no tempo da música
        payload_nohit = {
            "status": "no-hit",
            "elapsed_seconds": round(time.time() - start_t, 3),
            "hit": False,
            "best_pitch_hz": best["best_pitch_hz"],
            "best_error_cents_abs": best["best_error_cents_abs"],
            "best_volume_dbfs": best["best_volume_dbfs"],
            "nearest_note": best["nearest_note"],
            "target_pitch_hz": req.target_pitch_hz,
            "tolerance_cents": req.tolerance_cents,
            "min_volume_dbfs": req.min_volume_dbfs,
            "music_path": req.music_path,
        }
        save_result_json(req.resultado_path, payload_nohit)

        return StartResponse(
            status="no-hit-saved",
            message="Tempo esgotado sem acerto; melhor desempenho salvo no resultado.",
            elapsed_seconds=payload_nohit["elapsed_seconds"],
            hit=False,
            pitch_hz=best["best_pitch_hz"],
            error_cents=None,  # erro “bruto” não faz sentido sem referência de sinal atual
            volume_dbfs=best["best_volume_dbfs"] if best["best_volume_dbfs"] is not None else None,
            nearest_note=best["nearest_note"],
            resultado_path=req.resultado_path,
        )

    except sr.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Erro no microfone/áudio: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao iniciar: {e}")

# ================ Execução local ================
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
