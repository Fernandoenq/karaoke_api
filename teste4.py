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

# ===================== Metas (foco em PITCH e AFINAÇÃO) =====================
# Regras: valor medido >= alvo => OK
# - pitch_hz: garante que há voz com pitch "suficiente" (ex.: >=120Hz)
# - tuning_score: quanto "sobrou" dentro da tolerância (em cents). Quanto MAIOR, melhor.
#   Ex.: tolerância 20c; se erro=7c -> score=13; se erro=21c -> score=0.
TARGETS = {
    "pitch_hz": 90.0,        # pelo menos um pitch acima deste valor
    "tuning_score": 12.0,     # exige boa afinação (com TOL=20c, erro ≤ 4c)
}

# Parâmetros de afinação
TUNING_TOLERANCE_CENTS = 20.0   # tolerância para considerar "afinou" (±20 cents)

# ===================== Config de áudio =====================
SR = 16_000
CHANNELS = 1
BLOCK_MS = 20
WINDOW_MS = 250
HOP_MS = 250
BASELINE_SEC = 1.0
EPS = 1e-12

# ======= GATES contra falsos positivos =======
LOUDNESS_FLOOR_DBFS = -40.0     # altura da voz
MIN_SNR_DB = 30.0               # exige ambiente relativamente limpo
MAX_NOISE_ZCR = 0.15
VAD_FACTOR = 3.5
SALIENCE_MIN_DB = 14.0          # mais exigente que 10 dB (evita pitch "fantasma")

# ======= VITÓRIA: exigir estabilidade (streak) =======
# Com HOP_MS=250ms, 8 janelas ≈ 2s de estabilidade.
WIN_STREAK = 1

# ===================== Estado global =====================
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_win_event = threading.Event()
_state_lock = threading.Lock()

_is_running = False

# snapshots/estado
_last_metrics: Optional[dict] = None
_last_checks: Optional[dict] = None
_last_line: Optional[str] = None

_has_won = False
_win_metrics: Optional[dict] = None
_win_checks: Optional[dict] = None

# >>> Novo: pico de loudness da sessão
_max_loudness_dbfs: Optional[float] = None  # atualizado ao longo da sessão

# Fila de áudio do callback -> worker
_audio_q: Queue[np.ndarray] = Queue(maxsize=256)

# ===================== Utilitários musicais =====================
A4_HZ = 440.0
SEMITONE_LOG2 = 1.0 / 12.0

def rms_dbfs(x: np.ndarray):
    rms = float(np.sqrt(np.mean(x**2) + EPS))  # Calcula o RMS (Root Mean Square)
    db = 20.0 * np.log10(rms + EPS)  # Converte RMS para dBFS
    return rms, db


def _hz_to_note_name_stub(f0: float) -> str:
    if not f0 or f0 <= 0:
        return "—"
    A4 = A4_HZ
    n = int(round(12 * np.log2(f0 / A4)))
    notes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    name = notes[(n + 9) % 12]
    octave = 4 + ((n + 9) // 12)
    return f"{name}{octave}"

def _cents_error_to_nearest(f0: float) -> float | None:
    """Erro (em cents) do f0 para a nota igual-temperada mais próxima (valor absoluto)."""
    if f0 is None or f0 <= 0:
        return None
    n_float = 12.0 * np.log2(f0 / A4_HZ)
    n_round = round(n_float)
    f_ref = A4_HZ * (2.0 ** (n_round * SEMITONE_LOG2))
    cents = 1200.0 * np.log2(f0 / f_ref)
    return abs(float(cents))

def _tuning_score_from_error(abs_cents: float | None, tol_cents: float) -> float | None:
    """Score = quanto sobrou dentro da tolerância (>=0)."""
    if abs_cents is None:
        return None
    return float(max(0.0, tol_cents - abs_cents))

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
    """Mantido por causa do gate de ruído via ZCR."""
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
    """OK se value >= target para cada chave em TARGETS."""
    results = {}
    for key, target in TARGETS.items():
        val = metrics.get(key, None)
        ok = (val is not None) and (val >= target)
        results[key] = {"value": val, "target": target, "ok": ok}
    results["overall_ok"] = all(v["ok"] for v in results.values())
    return results

def peak_salience_db(x: np.ndarray, sr_hz: int, fmin=70.0, fmax=500.0) -> float:
    """Gate de saliência para evitar pitch fantasma."""
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

# ===================== Callback de áudio =====================
def _audio_callback(indata, frames, time_info, status):
    try:
        _audio_q.put_nowait(np.copy(indata[:, 0]))
    except:
        # Descarta um bloco para não travar se encher
        try:
            _audio_q.get_nowait()
            _audio_q.put_nowait(np.copy(indata[:, 0]))
        except:
            pass

# ===================== Loop de análise (worker) =====================
def analysis_loop(device: Optional[int | str] = None):
    global _is_running, _last_metrics, _last_checks, _last_line, _has_won, _win_metrics, _win_checks, _max_loudness_dbfs

    block_samples = max(1, int(SR * (BLOCK_MS / 1000.0)))
    window_samples = max(block_samples, int(SR * (WINDOW_MS / 1000.0)))
    hop_samples = max(1, int(SR * (HOP_MS / 1000.0)))

    ring = deque(maxlen=window_samples)
    got_baseline = False
    baseline_buf = []

    stream = sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=block_samples,
        device=device,
        callback=_audio_callback,
    )

    try:
        _stop_event.clear()
        stream.start()
        with _state_lock:
            _is_running = True

        # ======== baseline ========
        need_baseline = int(SR * BASELINE_SEC)
        while not _stop_event.is_set() and not got_baseline:
            try:
                chunk = _audio_q.get(timeout=0.2)
                baseline_buf.append(chunk)
                if sum(len(c) for c in baseline_buf) >= need_baseline:
                    noise = np.concatenate(baseline_buf)[:need_baseline]
                    noise_rms, noise_dbfs = rms_dbfs(noise)
                    got_baseline = True
                    print(f"Baseline ruído: RMS={noise_rms:.6f}, dBFS={noise_dbfs:.2f} dB")
                    print(
                        f"Metas (>=): {TARGETS} | SR={SR}Hz, WINDOW={WINDOW_MS}ms, HOP={HOP_MS}ms, "
                        f"VAD={VAD_FACTOR}x, Gates: floor={LOUDNESS_FLOOR_DBFS}dBFS, "
                        f"SNR>={MIN_SNR_DB}dB, ZCR<{MAX_NOISE_ZCR}, "
                        f"Sal>={SALIENCE_MIN_DB}dB, WIN_STREAK={WIN_STREAK}"
                    )
            except Empty:
                continue
        if not got_baseline:
            return

        samples_since_last = 0
        noise_rms, _ = rms_dbfs(np.concatenate(baseline_buf)[:need_baseline])

        # ======== loop principal ========
        win_streak = 0  # exige janelas consecutivas OK
        with _state_lock:
            _max_loudness_dbfs = None  # >>> zera o pico no início da sessão

        while not _stop_event.is_set():
            try:
                chunk = _audio_q.get(timeout=0.2)
            except Empty:
                continue

            ring.extend(chunk)
            samples_since_last += len(chunk)

            if len(ring) < window_samples or samples_since_last < hop_samples:
                continue

            x = np.frombuffer(np.array(ring, dtype=np.float32), dtype=np.float32)
            samples_since_last = 0

            # Métricas principais para gates
            rms, dbfs = rms_dbfs(x)
            snr_db = 20.0 * np.log10((rms + EPS) / (noise_rms + EPS))
            centroid, rolloff, zcr = spectral_features(x, SR)
            speaking = rms > (noise_rms * VAD_FACTOR)

            # ======== GATES ========
            is_silence = (dbfs < LOUDNESS_FLOOR_DBFS)
            snr_bad = (snr_db < MIN_SNR_DB)
            noise_like = (zcr > MAX_NOISE_ZCR and dbfs < (LOUDNESS_FLOOR_DBFS + 5.0))

            gates_fail = is_silence or snr_bad or noise_like or (not speaking)

            # >>> Atualiza pico ANTES do gate de saliência, mas somente se passou nos gates principais
            if not gates_fail:
                with _state_lock:
                    if (_max_loudness_dbfs is None) or (dbfs > _max_loudness_dbfs):
                        _max_loudness_dbfs = float(dbfs)

            if gates_fail:
                with _state_lock:
                    _last_metrics = {
                        "pitch_hz": None,
                        "tuning_error_cents": None,
                        "tuning_score": None,
                        "loudness_dbfs": float(dbfs),
                        "loudness_dbfs_peak": None if _max_loudness_dbfs is None else float(_max_loudness_dbfs),  # <<<
                    }
                    _last_checks = None
                    _last_line = (
                        f"[skip] L={dbfs:.1f}dBFS SNR={snr_db:.1f} ZCR={zcr:.3f} speaking={speaking}"
                    )
                win_streak = 0
                continue

            # ======== GATE de saliência espectral ========
            sal_db = peak_salience_db(x, SR)
            if sal_db < SALIENCE_MIN_DB:
                with _state_lock:
                    _last_metrics = {
                        "pitch_hz": None,
                        "tuning_error_cents": None,
                        "tuning_score": None,
                        "loudness_dbfs": float(dbfs),
                        "loudness_dbfs_peak": None if _max_loudness_dbfs is None else float(_max_loudness_dbfs),  # <<<
                    }
                    _last_checks = None
                    _last_line = f"[skip] salience={sal_db:.1f} dB"
                win_streak = 0
                continue

            # ======== Pitch + afinação ========
            f0 = estimate_pitch_fft(x, SR, fmin=70, fmax=500)
            err_cents = _cents_error_to_nearest(f0) if f0 else None
            tune_score = _tuning_score_from_error(err_cents, TUNING_TOLERANCE_CENTS)

            with _state_lock:
                metrics = {
                    "pitch_hz": None if f0 is None else float(f0),
                    "tuning_error_cents": None if err_cents is None else float(err_cents),
                    "tuning_score": None if tune_score is None else float(tune_score),
                    "loudness_dbfs": float(dbfs),
                    "loudness_dbfs_peak": None if _max_loudness_dbfs is None else float(_max_loudness_dbfs),  # <<<
                }
            checks = eval_goals(metrics)

            # Logs focados em pitch/afinação + loudness
            note_name = _hz_to_note_name_stub(f0) if f0 else "—"
            with _state_lock:
                peak_show = None if _max_loudness_dbfs is None else float(_max_loudness_dbfs)
            line = (
                f"[fala={speaking}] "
                f"Pitch={(f'{f0:.2f} Hz' if f0 else '—')} ({note_name}) | "
                f"Afinação: erro={(f'{err_cents:.1f} c' if err_cents is not None else '—')} | "
                f"Score={(f'{tune_score:.1f}' if tune_score is not None else '—')} | "
                f"Loudness={dbfs:.1f} dBFS (peak={peak_show:.1f} dBFS) | "
                f"Sal={sal_db:.1f} dB | "
                f"streak={win_streak}"
            )

            parts = []
            for k, r in checks.items():
                if k == "overall_ok":
                    continue
                val, tgt, ok = r["value"], r["target"], r["ok"]
                if val is None:
                    parts.append(f"{k}: ✗(None < {tgt})")
                else:
                    if k == "tuning_score":
                        parts.append(f"{k}: {'✓' if ok else '✗'}({val:.1f} ≥ {tgt})")
                    else:
                        parts.append(f"{k}: {'✓' if ok else '✗'}({val:.2f} ≥ {tgt})")
            goals_line = " | ".join(parts)
            overall = "OK" if checks["overall_ok"] else "FAIL"

            print(line)
            print(f"METAS [{overall}] -> {goals_line}\n")

            # Atualiza snapshots
            with _state_lock:
                _last_metrics = metrics
                _last_checks = checks
                _last_line = line

            # ======== Streak de vitória ========
            if checks["overall_ok"]:
                win_streak += 1
            else:
                win_streak = 0

            if (win_streak >= WIN_STREAK) and (not _has_won):
                with _state_lock:
                    _has_won = True
                    _win_metrics = metrics.copy()
                    _win_checks = checks.copy()
                _win_event.set()  # sinaliza vitória

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
app = FastAPI(title="Audio Metrics API (sounddevice)", version="3.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _snapshot_for_response():
    """Retorna um snapshot thread-safe do estado atual."""
    with _state_lock:
        return {
            "last_metrics": _last_metrics,
            "last_checks": _last_checks,
            "win_metrics": _win_metrics,
            "win_checks": _win_checks,
            "has_won": _has_won,
            "running": _is_running,
            "targets": TARGETS,
            "tuning_tolerance_cents": TUNING_TOLERANCE_CENTS,
            # informativo (opcional para debug no front, se quiser usar /status):
            "streak_required": WIN_STREAK,
            "salience_min_db": SALIENCE_MIN_DB,
        }

@app.get("/start")
def start_analysis(device: Optional[int | str] = None):
    """
    Inicia a captura e BLOQUEIA até:
      - ganhar (parabéns + métricas),
      - ou receber /stop antes de ganhar (não foi dessa vez + últimas métrricas).
    """
    global _worker_thread, _has_won, _win_metrics, _win_checks, _max_loudness_dbfs
    # limpar estado de sessão anterior
    with _state_lock:
        _has_won = False
        _win_metrics = None
        _win_checks = None
        _max_loudness_dbfs = None  # >>> zera pico ao iniciar
    _win_event.clear()

    if not (_worker_thread and _worker_thread.is_alive()):
        _stop_event.clear()
        _worker_thread = threading.Thread(target=analysis_loop, kwargs={"device": device}, daemon=True)
        _worker_thread.start()

    # Loop de espera: vitória ou stop
    while True:
        if _stop_event.is_set():
            # Stop antes de ganhar
            snap = _snapshot_for_response()
            return {
                "ok": True,
                "status": "nao_foi_dessa_vez",
                "message": "Não foi dessa vez.",
                "metrics": snap["last_metrics"],
                "checks": snap["last_checks"],
                "targets": snap["targets"],
                "tuning_tolerance_cents": snap["tuning_tolerance_cents"],
            }
        time.sleep(0.05)

@app.get("/stop")
def stop_analysis():
    """
    Para a análise imediatamente.
    - Se já tinha vitória: devolve 'parabéns' + métricas da vitória.
    - Caso contrário: 'não foi dessa vez' + últimas métricas.
    """
    _stop_event.set()

    # Espera a thread encerrar rapidamente (opcional)
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=1.0)

    snap = _snapshot_for_response()
    if snap["has_won"] and snap["win_metrics"] is not None:
        return {
            "ok": True,
            "status": "parabens",
            "message": "Parabéns! Você afinou dentro da meta.",
            "metrics": snap["win_metrics"],
            "checks": snap["win_checks"],
            "targets": snap["targets"],
            "tuning_tolerance_cents": snap["tuning_tolerance_cents"],
        }
    else:
        return {
            "ok": True,
            "status": "nao_foi_dessa_vez",
            "message": "Não foi dessa vez.",
            "metrics": snap["last_metrics"],
            "checks": snap["last_checks"],
            "targets": snap["targets"],
            "tuning_tolerance_cents": snap["tuning_tolerance_cents"],
        }

@app.get("/status")
def status():
    """Status rápido (não bloqueante)."""
    return _snapshot_for_response()
