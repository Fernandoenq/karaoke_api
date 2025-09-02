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

# ===================== Metas =====================

# Meta visual de barra (0..150). Exige nível mínimo de entrada
TARGETS_UI = {
    "input_level": 140
}


# Alvos em dB SPL (estimados) — ajuste à sua experiência de jogo
TARGETS_SPL = {
  
}

# Mantemos dBFS só para gates e depuração (não para "parabéns")
# (0 dBFS é o teto, valores são negativos; quanto mais perto de 0, mais alto)
TARGETS_DBFS_DEBUG = {
    "loudness_dbfs": -18.0,
    "loudness_dbfs_peak": -12.0,
}

# ===================== Config de áudio =====================
SR = 16_000        # taxa de amostragem (Hz). 16k = suficiente p/ voz; ↑ mais fiel, ↓ mais leve
CHANNELS = 1       # nº de canais. 1 = mono (voz). 2 = estéreo. Use mono p/ simplificar
BLOCK_MS = 20      # duração de cada bloco de captura (ms). ↓ menor latência, ↑ mais CPU
WINDOW_MS = 250    # janela de análise (ms). ↑ mais estável, ↓ responde mais devagar
HOP_MS = 300       # passo entre análises (ms). < WINDOW = mais frequente, > WINDOW = menos sensível
BASELINE_SEC = 1.0 # tempo inicial (s) para medir ruído de fundo. ↑ pega baseline mais confiável
EPS = 1e-12        # epsilon para evitar divisão por zero/log(0). Não precisa mexer


# ======= GATES contra falsos positivos (em dBFS) =======
LOUDNESS_FLOOR_DBFS = -42.0  # suba (ex.: -45) para aceitar janelas mais fracas
MIN_SNR_DB = 24.0            # diminua (ex.: 18) p/ ambientes barulhentos
VAD_FACTOR = 3.0             # diminua (ex.: 2.0) p/ detectar fala mais fraca
MAX_NOISE_ZCR = 0.15         # aumente (ex.: 0.20) p/ ser menos rígido com ruído


# ======= VITÓRIA: exigir estabilidade (streak) =======
WIN_STREAK = 1

# ======= Âncora para SPL sem calibração =======
# Interpretamos o dBFS do baseline como este SPL típico.
ASSUMED_BASELINE_SPL_DB = 40.0  # ajuste: 40–50 dB SPL para ambiente “calmo”, 50–60 em “barulhento”

# ===================== Estado global =====================
_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_win_event = threading.Event()
_state_lock = threading.Lock()

_is_running = False

_last_metrics: Optional[dict] = None
_last_checks: Optional[dict] = None
_last_line: Optional[str] = None

_has_won = False
_win_metrics: Optional[dict] = None
_win_checks: Optional[dict] = None

_last_valid_metrics: Optional[dict] = None
_last_valid_checks: Optional[dict] = None
_last_valid_line: Optional[str] = None

_best_metrics_so_far: Optional[dict] = None
_best_checks_so_far: Optional[dict] = None

_max_loudness_dbfs: Optional[float] = None

_audio_q: Queue[np.ndarray] = Queue(maxsize=256)

# ===================== Utilitários =====================


def eval_goals_ui(metrics: dict) -> dict:
    results = {}
    for key, target in TARGETS_UI.items():
        val = metrics.get(key, None)
        ok = (val is not None) and (val >= target)
        results[key] = {"value": val, "target": target, "ok": ok}
    results["overall_ok"] = all(v["ok"] for v in results.values())
    return results



def rms_dbfs(x: np.ndarray):
    rms = float(np.sqrt(np.mean(x**2) + EPS))
    db = 20.0 * np.log10(rms + EPS)
    return rms, db

def spectral_features(x: np.ndarray, sr_hz: int):
    n = int(2 ** np.ceil(np.log2(len(x))))
    spec = np.fft.rfft(x * np.hanning(len(x)), n=n)
    mag = np.abs(spec) + EPS
    freqs = np.fft.rfftfreq(n, d=1.0/sr_hz)
    # centroid/rolloff não são usados nos checks; só ZCR aqui
    zc = np.where(np.diff(np.signbit(x)))[0]
    zcr = float(len(zc) / max(len(x)-1, 1))
    return None, None, zcr

def dbfs_to_spl_est(dbfs_now: float, dbfs_baseline: float, baseline_spl_assumed: float) -> float:
    """
    Conversão sem calibração:
    SPL_est ≈ (dbfs_now - dbfs_baseline) + baseline_spl_assumed
    """
    return float((dbfs_now - dbfs_baseline) + baseline_spl_assumed)

def eval_goals_spl(metrics: dict) -> dict:
    """
    Avalia metas em dB SPL (estimado).
    OK se value >= target.
    """
    results = {}
    for key, target in TARGETS_SPL.items():
        val = metrics.get(key, None)
        ok = (val is not None) and (val >= target)
        results[key] = {"value": val, "target": target, "ok": ok}
    results["overall_ok"] = all(v["ok"] for v in results.values())
    return results

# ===================== Callback de áudio =====================
def _audio_callback(indata, frames, time_info, status):
    try:
        _audio_q.put_nowait(np.copy(indata[:, 0]))
    except:
        try:
            _audio_q.get_nowait()
            _audio_q.put_nowait(np.copy(indata[:, 0]))
        except:
            pass




####


def level_meter_value(dbfs: float, floor: float = -60.0, ceil: float = 0.0, max_ui: int = 150) -> int:
    """
    Converte dBFS para um valor 0..max_ui (simula a barrinha do microfone).
    floor: ruído de fundo (mínimo)
    ceil: teto (0 dBFS digital)
    """
    norm = (dbfs - floor) / (ceil - floor)
    return int(max(0, min(max_ui, norm * max_ui)))




# ===================== Loop de análise (worker) =====================
def analysis_loop(device: Optional[int | str] = None):
    global _is_running, _last_metrics, _last_checks, _last_line, _has_won
    global _win_metrics, _win_checks, _max_loudness_dbfs
    global _best_metrics_so_far, _best_checks_so_far

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
                    print(f"Baseline ruído: dBFS={noise_dbfs:.2f} (assumido ≈ {ASSUMED_BASELINE_SPL_DB:.1f} dB SPL)")
            except Empty:
                continue
        if not got_baseline:
            return

        samples_since_last = 0
        _, noise_dbfs = rms_dbfs(np.concatenate(baseline_buf)[:need_baseline])

        win_streak = 0
        with _state_lock:
            _max_loudness_dbfs = None

        # ======== loop principal ========
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

            # -------- Métricas principais --------
            rms, dbfs = rms_dbfs(x)
            if dbfs > 0:
                dbfs = 0.0
            if dbfs < -120:
                dbfs = -120.0

            snr_db = 20.0 * np.log10((rms + EPS) / (10**(noise_dbfs/20.0) + EPS))
            _, _, zcr = spectral_features(x, SR)
            speaking = rms > (10**(noise_dbfs/20.0) * VAD_FACTOR)

            # Gates (em dBFS)
            is_silence = (dbfs < LOUDNESS_FLOOR_DBFS)
            snr_bad = (snr_db < MIN_SNR_DB)
            noise_like = (zcr > MAX_NOISE_ZCR and dbfs < (LOUDNESS_FLOOR_DBFS + 5.0))
            gates_fail = is_silence or snr_bad or noise_like or (not speaking)

            # Atualiza pico dBFS
            if not gates_fail:
                with _state_lock:
                    if (_max_loudness_dbfs is None) or (dbfs > _max_loudness_dbfs):
                        _max_loudness_dbfs = float(dbfs)

            # SPL estimado
            spl_est = dbfs_to_spl_est(dbfs, noise_dbfs, ASSUMED_BASELINE_SPL_DB)
            spl_peak_est = None
            with _state_lock:
                if _max_loudness_dbfs is not None:
                    spl_peak_est = dbfs_to_spl_est(_max_loudness_dbfs, noise_dbfs, ASSUMED_BASELINE_SPL_DB)

            # -------- montar métricas --------
            metrics = {
                "loudness_dbfs": float(dbfs),
                "loudness_dbfs_peak": None if _max_loudness_dbfs is None else float(_max_loudness_dbfs),
                "loudness_db_spl": float(spl_est),
                "loudness_db_spl_peak": None if spl_peak_est is None else float(spl_peak_est),
                "input_level": level_meter_value(dbfs)  # 👈 barra do microfone
            }

            # -------- checar metas --------
            checks_spl = eval_goals_spl(metrics)
            checks_ui = eval_goals_ui(metrics)

            # mescla os dois
            checks = {**checks_spl, **checks_ui}
            checks["overall_ok"] = checks_spl["overall_ok"] and checks_ui["overall_ok"]

            # -------- salvar estado --------
            with _state_lock:
                _last_metrics = metrics
                _last_checks = checks
                _last_line = f"SPL={spl_est:.1f} dB | input_level={metrics['input_level']} | streak={win_streak}"

                _last_valid_metrics = metrics.copy()
                _last_valid_checks = checks.copy()
                _last_valid_line = _last_line

            # Vitória se passou em todas as metas
            if checks["overall_ok"]:
                win_streak += 1
            else:
                win_streak = 0

            if (win_streak >= WIN_STREAK) and (not _has_won):
                with _state_lock:
                    _has_won = True
                    _win_metrics = metrics.copy()
                    _win_checks = checks.copy()
                _win_event.set()

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
        try:
            while True:
                _audio_q.get_nowait()
        except Empty:
            pass

# ===================== API =====================
app = FastAPI(title="Audio Loudness API (SPL estimate)", version="4.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def _snapshot_for_response():
    with _state_lock:
        return {
            "last_metrics": _last_metrics,
            "last_checks": _last_checks,
            "last_valid_metrics": _last_valid_metrics,
            "last_valid_checks": _last_valid_checks,
            "best_metrics_so_far": _best_metrics_so_far,
            "best_checks_so_far": _best_checks_so_far,
            "win_metrics": _win_metrics,
            "win_checks": _win_checks,
            "has_won": _has_won,
            "running": _is_running,
            "targets_spl": TARGETS_SPL,
            "targets_dbfs_debug": TARGETS_DBFS_DEBUG,
            "assumed_baseline_spl_db": ASSUMED_BASELINE_SPL_DB,
            "streak_required": WIN_STREAK,
            "units": {
                "spl": "dB SPL (estimado)",
                "dbfs": "dBFS (relativo ao teto digital 0 dBFS)"
            },
        }

@app.get("/start")
def start_analysis(device: Optional[int | str] = None):
    """
    BLOQUEIA até ganhar (pelas metas SPL) ou receber /stop.
    """
    global _worker_thread, _has_won, _win_metrics, _win_checks, _max_loudness_dbfs, _last_valid_metrics, _last_valid_checks, _best_metrics_so_far, _best_checks_so_far
    with _state_lock:
        _has_won = False
        _win_metrics = None
        _win_checks = None
        _max_loudness_dbfs = None
        _last_valid_metrics = None
        _last_valid_checks = None
        _best_metrics_so_far = None
        _best_checks_so_far = None
    _win_event.clear()

    if not (_worker_thread and _worker_thread.is_alive()):
        _stop_event.clear()
        _worker_thread = threading.Thread(target=analysis_loop, kwargs={"device": device}, daemon=True)
        _worker_thread.start()

    while True:
        if _win_event.is_set():
            snap = _snapshot_for_response()
            return {
                "ok": True,
                "status": "parabens",
                "message": "Parabéns! Você atingiu o nível de volume (SPL) da meta.",
                "metrics": snap["win_metrics"],
                "checks": snap["win_checks"],
                "targets_spl": snap["targets_spl"],
                "assumed_baseline_spl_db": snap["assumed_baseline_spl_db"],
                "units": snap["units"],
            }
        if _stop_event.is_set():
            snap = _snapshot_for_response()
            return {
                "ok": True,
                "status": "nao_foi_dessa_vez",
                "message": "Não foi dessa vez.",
                "metrics": snap["last_metrics"],
                "checks": snap["last_checks"],
                "targets_spl": snap["targets_spl"],
                "assumed_baseline_spl_db": snap["assumed_baseline_spl_db"],
                "units": snap["units"],
            }
        time.sleep(0.05)

@app.get("/stop")
def stop_analysis():
    """
    Para imediatamente. Se ganhou, retorna vitória; senão, melhor tentativa (SPL).
    """
    _stop_event.set()

    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=1.0)

    snap = _snapshot_for_response()

    if snap["has_won"] and snap["win_metrics"] is not None:
        return {
            "ok": True,
            "status": "parabens",
            "message": "Parabéns! Você atingiu o nível de volume (SPL) da meta.",
            "metrics": snap["win_metrics"],
            "checks": snap["win_checks"],
            "targets_spl": snap["targets_spl"],
            "assumed_baseline_spl_db": snap["assumed_baseline_spl_db"],
            "units": snap["units"],
        }

    m = snap.get("best_metrics_so_far") or snap.get("last_valid_metrics") or snap.get("last_metrics")
    c = snap.get("best_checks_so_far") or snap.get("last_valid_checks") or snap.get("last_checks")

    return {
        "ok": True,
        "status": "nao_foi_dessa_vez",
        "message": "Não foi dessa vez. Aqui está sua melhor tentativa (SPL estimado).",
        "metrics": m,
        "checks": c,
        "targets_spl": snap["targets_spl"],
        "assumed_baseline_spl_db": snap["assumed_baseline_spl_db"],
        "units": snap["units"],
    }

@app.get("/status")
def status():
    """Status rápido."""
    return _snapshot_for_response()
