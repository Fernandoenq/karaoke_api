# app.py
from __future__ import annotations

import threading
import time
from typing import Optional
from collections import deque
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ===================== Metas (atingiu OU ultrapassou) =====================
# Regras fáceis p/ testar: valor medido >= alvo => OK
TARGETS = {
    "pitch_hz": 120,
    "loudness_dbfs": -35.0,
    "snr_db": 10.0,
    "centroid_hz": 1200.0,
    "rolloff85_hz": 3000.0,
    "zcr": 0.15,
}

# ===================== Config de áudio leve (baixo custo) =====================
SR = 16_000                # 16 kHz suficiente p/ voz (menos CPU que 44.1/48)
CHANNELS = 1               # mono
BLOCK_MS = 20              # bloco do callback (~20ms) -> responsivo
WINDOW_MS = 250            # janela de análise (~0.25s)
HOP_MS = 250               # hop = janela (atualização ~4x/s)
BASELINE_SEC = 1.0         # tempo p/ medir ruído inicial
EPS = 1e-12

# ======= GATES contra falsos positivos =======
LOUDNESS_FLOOR_DBFS = -50.0   # abaixo disso, trate como silêncio
MIN_SNR_DB = 14.0              # SNR mínimo para considerar voz
MAX_NOISE_ZCR = 0.15          # ZCR alto + nível baixo => ruído
VAD_FACTOR = 2.0              # fator de VAD (energia vs ruído). 1.5 é mais rigoroso que 1.2

# ===================== Estado global do worker =====================
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_state_lock = threading.Lock()
_is_running = False

# Fila de áudio do callback -> worker (mantém leve o callback)
_audio_q: Queue[np.ndarray] = Queue(maxsize=int((SR * 2) / (SR * BLOCK_MS / 1000 + 1)))  # ~2s de buffer

def _hz_to_note_name_stub(f0: float) -> str:
    # Opcional: só p/ debug mais amigável (não é usado nas metas)
    if not f0 or f0 <= 0:
        return "—"
    A4 = 440.0
    n = int(round(12 * np.log2(f0 / A4)))
    notes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    name = notes[(n + 9) % 12]  # mapeia A->A, etc.
    octave = 4 + ((n + 9) // 12)
    return f"{name}{octave}"

# ===================== DSP helpers =====================
def estimate_pitch_fft(x: np.ndarray, sr_hz: int, fmin=70.0, fmax=500.0) -> float | None:
    x = x - np.mean(x)
    if np.max(np.abs(x)) < 1e-6:
        return None
    w = np.hanning(len(x))
    xw = x * w
    n = int(2 ** np.ceil(np.log2(len(xw))))
    spec = np.fft.rfft(xw, n=n)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0/sr_hz)

    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size < 3:
        return None
    sub_mag = mag[idx]
    k = int(np.argmax(sub_mag))
    k_global = idx[k]

    if 1 <= k_global < len(mag) - 1:
        alpha = mag[k_global - 1]
        beta  = mag[k_global]
        gamma = mag[k_global + 1]
        denom = (alpha - 2*beta + gamma)
        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
        refined_bin = k_global + float(delta)
        f0 = refined_bin * (sr_hz / n)
    else:
        f0 = freqs[k_global]
    return float(f0)

def spectral_features(x: np.ndarray, sr_hz: int):
    n = int(2 ** np.ceil(np.log2(len(x))))
    spec = np.fft.rfft(x * np.hanning(len(x)), n=n)
    mag = np.abs(spec) + EPS
    freqs = np.fft.rfftfreq(n, d=1.0/sr_hz)

    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    cumulative = np.cumsum(mag)
    cutoff = 0.85 * cumulative[-1]
    k_roll = int(np.searchsorted(cumulative, cutoff))
    rolloff = float(freqs[min(k_roll, len(freqs)-1)])

    zc = np.where(np.diff(np.signbit(x)))[0]
    zcr = float(len(zc) / max(len(x)-1, 1))
    return centroid, rolloff, zcr

def rms_dbfs(x: np.ndarray):
    rms = float(np.sqrt(np.mean(x**2) + EPS))
    db = 20.0 * np.log10(rms + EPS)
    return rms, db

def eval_goals(metrics: dict) -> dict:
    results = {}
    for key, target in TARGETS.items():
        val = metrics.get(key, None)
        ok = (val is not None) and (val >= target)
        results[key] = {"value": val, "target": target, "ok": ok}
    results["overall_ok"] = all(v["ok"] for v in results.values())
    return results

def peak_salience_db(x: np.ndarray, sr_hz: int, fmin=70.0, fmax=500.0) -> float:
    """Mede 'salientez' do pico (pico vs mediana) no subespectro de interesse."""
    n = int(2 ** np.ceil(np.log2(len(x))))
    spec = np.fft.rfft(x * np.hanning(len(x)), n=n)
    mag = np.abs(spec) + EPS
    freqs = np.fft.rfftfreq(n, d=1.0/sr_hz)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size < 8:
        return 0.0
    sub = mag[idx]
    peak = float(np.max(sub))
    med = float(np.median(sub) + EPS)
    return 20.0 * np.log10(peak / med)

# ===================== Callback de áudio (super leve) =====================
def _audio_callback(indata, frames, time_info, status):
    if status:
        # Evite prints aqui para não pesar; logs no worker se quiser
        pass
    # Copia o bloco mono (float32 [-1,1]) e empilha na fila
    try:
        _audio_q.put_nowait(np.copy(indata[:, 0]))
    except:
        # se a fila encher, descarte silenciosamente (preferível a travar)
        try:
            _audio_q.get_nowait()
            _audio_q.put_nowait(np.copy(indata[:, 0]))
        except:
            pass

# ===================== Loop de análise (worker) =====================
def analysis_loop(device: Optional[int | str] = None):
    global _is_running

    block_samples = max(1, int(SR * (BLOCK_MS / 1000.0)))
    window_samples = max(block_samples, int(SR * (WINDOW_MS / 1000.0)))
    hop_samples = max(1, int(SR * (HOP_MS / 1000.0)))

    ring = deque(maxlen=window_samples)
    got_baseline = False
    baseline_buf = []

    # Abra o stream de entrada contínuo
    stream = sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_samples,   # 0 = "ideal do backend"; usar valor fixo ajuda previsibilidade
        device=device,             # None = padrão; pode passar índice ou nome
        callback=_audio_callback,
    )

    try:
        _stop_event.clear()
        stream.start()
        with _state_lock:
            _is_running = True

        # ======== Baseline de ruído ========
        need_baseline_samples = int(SR * BASELINE_SEC)
        while not _stop_event.is_set() and not got_baseline:
            try:
                chunk = _audio_q.get(timeout=0.2)
                baseline_buf.append(chunk)
                if sum(len(c) for c in baseline_buf) >= need_baseline_samples:
                    noise = np.concatenate(baseline_buf)[:need_baseline_samples]
                    noise_rms, noise_dbfs = rms_dbfs(noise)
                    got_baseline = True
                    print(f"Baseline ruído: RMS={noise_rms:.6f}, dBFS={noise_dbfs:.2f} dB")
                    print(
                        f"Metas (>=): {TARGETS} | SR={SR}Hz, WINDOW={WINDOW_MS}ms, HOP={HOP_MS}ms, "
                        f"VAD={VAD_FACTOR}x, Gates: floor={LOUDNESS_FLOOR_DBFS}dBFS, SNR>={MIN_SNR_DB}dB, ZCR<{MAX_NOISE_ZCR}"
                    )
            except Empty:
                continue

        if not got_baseline:
            return

        # ======== Loop principal ========
        samples_since_last = 0
        noise_rms, _ = rms_dbfs(np.concatenate(baseline_buf)[:need_baseline_samples])

        while not _stop_event.is_set():
            try:
                chunk = _audio_q.get(timeout=0.2)
            except Empty:
                continue

            ring.extend(chunk)
            samples_since_last += len(chunk)

            if len(ring) < window_samples:
                continue  # ainda não temos janela cheia

            if samples_since_last < hop_samples:
                continue  # só processa a cada hop

            # Coleta janela corrente
            x = np.frombuffer(np.array(ring, dtype=np.float32), dtype=np.float32)  # cópia
            samples_since_last = 0

            # Métricas (exceto pitch; só vamos calcular se passar nos gates)
            rms, dbfs = rms_dbfs(x)
            snr_db = 20.0 * np.log10((rms + EPS) / (noise_rms + EPS))
            centroid, rolloff, zcr = spectral_features(x, SR)
            speaking = rms > (noise_rms * VAD_FACTOR)

            # ======== GATES: silêncio/ruído/VAD ========
            is_silence = (dbfs < LOUDNESS_FLOOR_DBFS)
            snr_bad = (snr_db < MIN_SNR_DB)
            noise_like = (zcr > MAX_NOISE_ZCR and dbfs < (LOUDNESS_FLOOR_DBFS + 5.0))
            not_speaking = not speaking
            if is_silence or snr_bad or noise_like or not_speaking:
                # Opcional: log de debug
                # print(f"[skip] L={dbfs:.1f}dBFS SNR={snr_db:.1f} ZCR={zcr:.3f} speaking={speaking}")
                continue

            # ======== GATE de salientez espectral p/ evitar pitch fantasma ========
            sal_db = peak_salience_db(x, SR)  # pico vs mediana no subespectro 70–500 Hz
            if sal_db < 10.0:  # ajuste 8–12 dB conforme ambiente
                # print(f"[skip] salience={sal_db:.1f} dB")
                continue

            # Agora sim estime o pitch (já passou nos filtros)
            f0 = estimate_pitch_fft(x, SR, fmin=70, fmax=500)

            metrics = {
                "pitch_hz": None if f0 is None else float(f0),
                "loudness_dbfs": float(dbfs),
                "snr_db": float(snr_db),
                "centroid_hz": float(centroid),
                "rolloff85_hz": float(rolloff),
                "zcr": float(zcr),
            }
            checks = eval_goals(metrics)

            # Prints (duas linhas)
            line = (
                f"[fala={speaking}] "
                f"Pitch={(f'{f0:.2f} Hz' if f0 else '—')} ({_hz_to_note_name_stub(f0) if f0 else '—'}) | "
                f"Loudness={dbfs:.2f} dBFS | SNR={snr_db:.1f} dB | "
                f"Centroid={centroid:.1f} Hz | Rolloff(85%)={rolloff:.0f} Hz | ZCR={zcr:.3f} | "
                f"Sal={sal_db:.1f} dB"
            )
            goals_str_parts = []
            for k, r in checks.items():
                if k == "overall_ok":
                    continue
                val, tgt, ok = r["value"], r["target"], r["ok"]
                if val is None:
                    goals_str_parts.append(f"{k}: ✗(None < {tgt})")
                else:
                    goals_str_parts.append(f"{k}: {'✓' if ok else '✗'}({val:.2f} ≥ {tgt})")
            goals_str = " | ".join(goals_str_parts)
            overall = "OK" if checks["overall_ok"] else "FAIL"

            print(line)
            print(f"METAS [{overall}] -> {goals_str}\n")

    except Exception as e:
        print(f"Erro no stream: {e}")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        with _state_lock:
            _is_running = False
        # esvazia fila
        try:
            while True:
                _audio_q.get_nowait()
        except Empty:
            pass

# ===================== API =====================
app = FastAPI(title="Audio Metrics API (sounddevice)", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/start")
def start_analysis(device: Optional[int | str] = None):
    """Inicia captura contínua (prints no console). Parâmetro opcional ?device=<índice ou nome>."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return {"ok": True, "status": "already_running"}

    _stop_event.clear()
    _worker_thread = threading.Thread(target=analysis_loop, kwargs={"device": device}, daemon=True)
    _worker_thread.start()
    return {
        "ok": True, "status": "started",
        "device": device,
        "targets": TARGETS,
        "audio": {
            "sr": SR,
            "block_ms": BLOCK_MS,
            "window_ms": WINDOW_MS,
            "hop_ms": HOP_MS,
            "vad_factor": VAD_FACTOR,
            "gates": {
                "loudness_floor_dbfs": LOUDNESS_FLOOR_DBFS,
                "min_snr_db": MIN_SNR_DB,
                "max_noise_zcr": MAX_NOISE_ZCR,
                "salience_min_db": 10.0,
            },
        },
    }

@app.get("/stop")
def stop_analysis():
    """Para a análise imediatamente."""
    _stop_event.set()
    return {"ok": True, "status": "stopping"}

@app.get("/status")
def status():
    """Retorna se a análise está em execução."""
    with _state_lock:
        return {"running": _is_running}
