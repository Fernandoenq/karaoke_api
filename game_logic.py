import os
import time
import threading
import json
import queue
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init as colorama_init

# ============================================================
#  Versão sem AUBIO: usa LIBROSA (YIN) para detecção de pitch
#  Requisitos:
#      pip install librosa sounddevice soundfile colorama numpy
#  Compatível com Windows + Python 3.10
# ============================================================
try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

colorama_init(autoreset=True)

# ======================= CONFIG PADRÃO DO JOGO =======================
TARGET_PITCH_HZ = 440        # ~ A4 (padrão)
TOLERANCE_HZ    = 40.0
COUNTDOWN_S     = 3          # 3,2,1 e começa

# Faixa típica de voz (ajuste se quiser outro instrumento/voz)
FMIN_HZ = 70.0
FMAX_HZ = 800.0

# ===== PARÂMETROS DE AVALIAÇÃO RIGOROSA (em cents/segundos) =====
WIN_RATIO_MIN          = 0.75   # proporção de acertos mínima (antes era 0.6)
MIN_VOICED_SECONDS     = 2.0    # tempo mínimo de voz válida antes de permitir vitória
MIN_CONTIG_STREAK_S    = 1.0    # tempo contínuo mínimo "em cima" do alvo
MAX_JITTER_CENTS       = 35.0   # estabilidade: desvio padrão máx. dos f0 em cents
MAX_MEDIAN_DEV_CENTS   = 25.0   # quão perto do alvo (mediana) precisa estar
MIN_RMS                = 0.02   # energia mínima por frame para considerar "voz"
USE_CENTS_HIT          = True   # usa comparação em cents (recomendado)


# ======================= ESTADO & RESULTADO ==========================
class SessionState(Enum):
    idle = "idle"
    countdown = "countdown"
    running = "running"
    finished = "finished"
    stopping = "stopping"

@dataclass
class SessionResult:
    state: SessionState = SessionState.finished
    result: Optional[str] = None           # "ganhou" | "perdeu"
    hit_ratio: Optional[float] = None
    max_streak_s: Optional[float] = None
    target_hz: Optional[float] = None
    tolerance_hz: Optional[float] = None
    achieved_pitch_hz: Optional[float] = None
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None

# ======================= HELPERS DE LOG BONITO =======================

def _hz_to_cents_ratio(f_hz: float, ref_hz: float) -> float:
    """Retorna diferença em cents entre f_hz e ref_hz."""
    if f_hz <= 0 or ref_hz <= 0:
        return float('inf')
    return 1200.0 * np.log2(f_hz / ref_hz)

def _nearest_octave_ref(f_hz: float, target_hz: float) -> float:
    """Ajusta target para a oitava mais próxima de f_hz (evita dar win só por oitava errada)."""
    if f_hz <= 0 or target_hz <= 0:
        return target_hz
    k = round(np.log2(f_hz / target_hz))  # passo em oitavas
    return target_hz * (2.0 ** k)

def _is_hit_strict_cents(f_hz: float, target_hz: float, cents_tol: float) -> bool:
    """Acerto se diferença em cents ao target (na oitava mais próxima) for <= tolerância."""
    if f_hz <= 0:
        return False
    ref = _nearest_octave_ref(f_hz, target_hz)
    dev_cents = abs(_hz_to_cents_ratio(f_hz, ref))
    return dev_cents <= cents_tol

def _rms(block: np.ndarray) -> float:
    """Energia RMS simples do frame mono (float32)."""
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))


def _is_hit(pitch_hz: float, target_hz: float, tol_hz: float) -> bool:
    if USE_CENTS_HIT:
        # ~ 40 Hz perto de A4 ≈ 157 cents; vamos usar 35–40c como padrão mais musical
        return _is_hit_strict_cents(pitch_hz, target_hz, cents_tol=35.0)
    # legacy em Hz com oitavas:
    for k in (-2, -1, 0, 1, 2):
        t = target_hz * (2 ** k)
        if abs(pitch_hz - t) <= tol_hz:
            return True
    return False


def _banner(title: str) -> str:
    bar = "═" * 60
    return f"\n{Fore.CYAN}{bar}\n{title.center(60)}\n{bar}{Style.RESET_ALL}"

def _kv(key: str, val: Any, color=Fore.WHITE) -> str:
    return f"{color}{key:<18}{Style.RESET_ALL}: {val}"

def _ok(msg: str) -> str:
    return f"{Fore.GREEN}{msg}{Style.RESET_ALL}"

def _warn(msg: str) -> str:
    return f"{Fore.YELLOW}{msg}{Style.RESET_ALL}"

def _err(msg: str) -> str:
    return f"{Fore.RED}{msg}{Style.RESET_ALL}"

# ======================= DISPOSITIVOS ================================
def list_audio_devices() -> Dict[str, Any]:
    """Retorna devices de áudio (entrada/saída) em tipos nativos."""
    info = sd.query_devices()
    defaults = sd.default.device  # [in, out]
    inputs: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []

    for idx, dev in enumerate(info):
        rec = dict(
            id=int(idx),
            name=str(dev.get("name", "")),
            max_input_channels=int(dev.get("max_input_channels", 0)),
            max_output_channels=int(dev.get("max_output_channels", 0)),
        )
        if rec["max_input_channels"] > 0:
            inputs.append(rec)
        if rec["max_output_channels"] > 0:
            outputs.append(rec)

    def _as_int(x):
        try:
            return int(x) if x is not None else None
        except Exception:
            return None

    return {"defaults": [_as_int(defaults[0]), _as_int(defaults[1])], "input": inputs, "output": outputs}

# ======================= DETECTOR DE PITCH (LIBROSA) =================
class LibrosaPitchDetector:
    """
    Detecção de pitch via YIN do librosa, em tempo real.
    Mantém um buffer deslizante de tamanho frame_length e,
    a cada hop, estima o f0 para a janela mais recente.
    """
    def __init__(
        self,
        sr: int,
        fmin: float = FMIN_HZ,
        fmax: float = FMAX_HZ,
        frame_length: int = 2048,
        hop_length: int = 512,
        periodicity_threshold: float = 0.1,  # quanto menor, mais sensível
    ):
        self.sr = int(sr)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.periodicity_threshold = float(periodicity_threshold)
        self._buffer = np.zeros(0, dtype=np.float32)

    def process_block(self, mono_block: np.ndarray) -> List[float]:
        """
        Recebe um bloco mono (float32) e retorna uma lista de f0 em Hz
        (um valor por hop consumido). Pode retornar lista vazia.
        """
        if mono_block.ndim != 1:
            mono_block = np.mean(mono_block, axis=-1)
        mono_block = mono_block.astype(np.float32, copy=False)

        # acumula
        self._buffer = np.concatenate([self._buffer, mono_block])

        f0_list: List[float] = []

        # Enquanto houver amostras suficientes para um frame:
        while len(self._buffer) >= self.frame_length:
            frame = self._buffer[-self.frame_length:]  # janela mais recente

            # O truque: hop_length == frame_length => 1 estimativa
            try:
                f0 = librosa.yin(
                    y=frame,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    frame_length=self.frame_length,
                    hop_length=self.frame_length,  # 1 saída
                    trough_threshold=self.periodicity_threshold,  # controle de "voicing"
                )
                hz = float(f0[0]) if np.isfinite(f0[0]) else 0.0
            except Exception:
                hz = 0.0

            f0_list.append(hz if (hz > 0.0 and np.isfinite(hz)) else 0.0)

            # avança o buffer em hop_length
            self._buffer = self._buffer[self.hop_length:]

        return f0_list

# ======================= ENGINE =====================================
class GameEngine:
    def __init__(self):
        self.state: SessionState = SessionState.idle
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._win_event = threading.Event()     # vitória antecipada
        self._live_q: "queue.Queue[str]" = queue.Queue(maxsize=1000)  # fila de eventos ao vivo

        # sessão
        self._music_path: Optional[str] = None
        self._audio_data: Optional[np.ndarray] = None
        self._sr: Optional[int] = None
        self._duration_s: float = 0.0

        # devices
        self._in_dev: Optional[int] = None
        self._out_dev: Optional[int] = None

        # progresso
        self._t_start: Optional[float] = None
        self._play_t: int = 0  # ponteiro de reprodução (frames)

        # resultado
        self._result: Optional[SessionResult] = None

    # ---------------------- eventos ao vivo (SSE) ---------------------
    def publish(self, kind: str, **payload):
        """Empilha um evento JSON na fila /live."""
        try:
            msg = {"t": time.time(), "kind": kind, **payload}
            s = json.dumps(msg, ensure_ascii=False)
            if self._live_q.full():
                try:
                    self._live_q.get_nowait()  # descarta o mais antigo
                except Exception:
                    pass
            self._live_q.put_nowait(s)
        except Exception:
            pass

    # ---------------------- utilidades de estado ----------------------
    def countdown_left(self) -> Optional[int]:
        if self.state != SessionState.countdown:
            return None
        return None

    def progress(self) -> float:
        if self.state != SessionState.running or self._t_start is None or self._duration_s <= 0:
            return 0.0
        elapsed = time.time() - self._t_start
        return float(max(0.0, min(1.0, elapsed / self._duration_s)))

    def status_message(self) -> str:
        if self.state == SessionState.idle:
            return "Pronto para iniciar"
        if self.state == SessionState.countdown:
            return "Contagem regressiva…"
        if self.state == SessionState.running:
            p = int(self.progress() * 100)
            return f"Executando ({p}%)"
        if self.state == SessionState.finished:
            return "Finalizado"
        if self.state == SessionState.stopping:
            return "Encerrando…"
        return "—"

    # ---------------------- ciclo de vida -----------------------------
    def prepare(self, music_path: Optional[str], in_dev: Optional[int], out_dev: Optional[int]) -> None:
        self._stop_flag.clear()
        self._win_event.clear()
        self._result = None
        self._music_path = music_path or "referencia_trecho.wav"  # fallback (relativo ao cwd)
        self._in_dev = in_dev
        self._out_dev = out_dev
        self._play_t = 0
        self.state = SessionState.countdown

        print(_banner("KARAOKÊ • PREPARE"))
        print(_kv("input dev", self._in_dev))
        print(_kv("output dev", self._out_dev))
        print(_kv("music_path", self._music_path))
        print(_kv("cwd", os.getcwd()))
        print(_kv("project_root", os.path.abspath(".")))
        self.publish("state", state="countdown", in_dev=self._in_dev, out_dev=self._out_dev, music=self._music_path)

    def stop(self) -> None:
        self._stop_flag.set()
        self.state = SessionState.stopping
        print(_warn("stop requested"))
        self.publish("state", state="stopping")

    def run_session_async(self) -> None:
        try:
            # -------- countdown ----------
            for i in range(COUNTDOWN_S, 0, -1):
                if self._stop_flag.is_set():
                    self.state = SessionState.finished
                    self.publish("finished", **self.result_summary())
                    return
                print(_kv("countdown", i, Fore.CYAN))
                self.publish("countdown", left=i)
                time.sleep(1)

            # -------- carregar áudio ----------
            print(_banner("KARAOKÊ • RESOLVE AUDIO"))
            resolved, err = self._resolve_music_path(self._music_path)
            print(_kv("resolve", _ok(f"ok: {resolved}") if err is None else _err(err)))
            print(_kv("cwd", os.getcwd()))
            if err:
                self._finish_with_error(err)
                return

            print(_banner("KARAOKÊ • AUDIO OK"))
            print(_kv("file", os.path.basename(resolved)))
            print(_kv("duration", f"{self._duration_s:.2f}s"))
            self.publish("audio_resolved", file=resolved, duration_s=self._duration_s)

            # -------- abrir streams ----------
            print(_banner("KARAOKÊ • OPEN STREAMS"))
            print(_kv("in dev", self._in_dev))
            print(_kv("out dev", self._out_dev))
            print(_kv("default", str(sd.default.device)))

            self.state = SessionState.running
            self._t_start = time.time()
            self.publish("state", state="running")

            print(_banner("KARAOKÊ • PLAYING & MEASURING"))

            # tocar + medir
            self._play_and_measure(resolved)

            # imprime resultado ao final
            self._finalize_and_print()

        except Exception as e:
            self._finish_with_error(str(e))

    # ---------------------- áudio & medição ---------------------------
    def _resolve_music_path(self, p: Optional[str]) -> Tuple[str, Optional[str]]:
        if not p:
            return "", "caminho de áudio vazio"

        # absoluto ou relativo
        candidate = p
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)

        try:
            data, sr = sf.read(candidate, dtype="float32", always_2d=True)
        except Exception as e:
            return p, f"erro ao abrir áudio: {e}"

        self._audio_data = data
        self._sr = int(sr)
        self._duration_s = float(len(data) / sr)
        return candidate, None

    # helper: vitória antecipada
    def _declare_win_now(self, hit_ratio: float, max_streak: float, frames_valid: int,
                         pitches: List[float]) -> None:
        achieved = float(np.median(pitches)) if len(pitches) else None
        self._result = SessionResult(
            state=SessionState.finished,
            result="ganhou",
            hit_ratio=float(hit_ratio),
            max_streak_s=float(max_streak),
            target_hz=float(TARGET_PITCH_HZ),
            tolerance_hz=float(TOLERANCE_HZ),
            achieved_pitch_hz=achieved,
            frames_valid=int(frames_valid),
            duration_s=float(self._duration_s),
            error=None
        )
        self.state = SessionState.finished
        self._win_event.set()
        self._stop_flag.set()
        self.publish("win", ratio=float(hit_ratio))

    def _play_and_measure(self, path: str) -> None:
        data = self._audio_data
        sr = self._sr
        if data is None or sr is None:
            self._finish_with_error("áudio não carregado")
            return

        # buffers maiores = CPU mais folgada = menos overflow
        blocksize = 4096  # tente 2048 ou 4096
        frame_len = blocksize
        hop_len = blocksize

        if not HAS_LIBROSA:
            print(_warn("librosa não encontrado — sessão seguirá sem medição de pitch."))
            self.publish("warn", message="librosa não encontrado")
        else:
            detector = LibrosaPitchDetector(
                sr=sr,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                frame_length=frame_len,
                hop_length=hop_len,
                periodicity_threshold=0.08,  # um pouco mais sensível
            )

        frames_hits = 0
        frames_valid = 0
        max_streak = 0.0
        current_streak = 0.0
        pitches: List[float] = []

        voiced_seconds = 0.0
        dev_cents_samples: List[float] = []
        rms_samples: List[float] = []

        WIN_THRESHOLD = WIN_RATIO_MIN  # critério de vitória
        last_emit = 0.0      # throttling (s)

        def out_cb(outdata, frames, time_info, status):
            del time_info
            if status:
                print(_warn(f"OUTPUT status: {status}"))
                self.publish("output_status", status=str(status))
            t = self._play_t
            end = min(t + frames, len(data))
            chunk = data[t:end]
            if len(chunk) < frames:
                pad = np.zeros((frames - len(chunk), data.shape[1]), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            outdata[:] = chunk
            self._play_t = end

        def in_cb(indata, frames, time_info, status):
            del time_info
            if status:
                print(_warn(f"INPUT status: {status}"))
                self.publish("input_status", status=str(status))

            nonlocal frames_hits, frames_valid, max_streak, current_streak, pitches, last_emit

            if not HAS_LIBROSA:
                return

            try:
                mono = np.mean(indata, axis=1).astype(np.float32)
                block_rms = _rms(np.asarray(indata, dtype=np.float32).reshape(-1))
                rms_samples.append(block_rms)

                f0_list = detector.process_block(mono)

                for f0 in f0_list:
                    ts = time.time()

                    # Só considera "válido" se há f0 e energia suficiente
                    if f0 > 0.0 and np.isfinite(f0) and block_rms >= MIN_RMS:
                        frames_valid += 1
                        pitches.append(float(f0))

                        # desvio em cents para a oitava mais próxima
                        ref = _nearest_octave_ref(f0, TARGET_PITCH_HZ)
                        dev_cents = _hz_to_cents_ratio(f0, ref)
                        dev_cents_samples.append(dev_cents)

                        hit = _is_hit(f0, TARGET_PITCH_HZ, TOLERANCE_HZ)
                        if hit:
                            frames_hits += 1
                            current_streak += frames / sr
                            if current_streak > max_streak:
                                max_streak = current_streak
                        else:
                            current_streak = 0.0

                        # acumula tempo de voz válida
                        voiced_seconds += frames / sr

                        # emitir evento ~10/s
                        if ts - last_emit >= 0.1:
                            ratio = frames_hits / max(1, frames_valid)
                            self.publish(
                                "pitch",
                                f0=float(f0),
                                hit=bool(hit),
                                ratio=float(ratio),
                                valid=int(frames_valid),
                                streak_s=float(current_streak),
                                rms=float(block_rms),
                                dev_cents=float(dev_cents),
                            )
                            last_emit = ts

                        # vitória antecipada MAIS RÍGIDA
                        if frames_valid >= 20:  # requer ao menos alguns frames válidos
                            ratio_now = frames_hits / max(1, frames_valid)
                            jitter = float(np.std(dev_cents_samples)) if len(dev_cents_samples) > 1 else 999.0
                            median_dev = float(np.median(np.abs(dev_cents_samples))) if dev_cents_samples else 999.0

                            if (
                                    ratio_now >= WIN_RATIO_MIN and
                                    voiced_seconds >= MIN_VOICED_SECONDS and
                                    max_streak >= MIN_CONTIG_STREAK_S and
                                    jitter <= MAX_JITTER_CENTS and
                                    median_dev <= MAX_MEDIAN_DEV_CENTS and
                                    not self._win_event.is_set()
                            ):
                                self._declare_win_now(ratio_now, max_streak, frames_valid, pitches)
                                return
                    else:
                        current_streak = 0.0

            except Exception as e:
                print(_err(f"in_cb error: {e}"))
                self.publish("error", where="in_cb", message=str(e))

        # abrir streams
        try:
            with sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    device=self._in_dev,
                    callback=in_cb,
                    dtype="float32",
                    blocksize=blocksize,  # <= usa o mesmo blocksize do processamento (4096)
                    latency="high",  # dá mais folga para CPU/driver; teste "low" depois
            ):

                start_time = time.time()
                while not self._stop_flag.is_set() and (time.time() - start_time) < self._duration_s:
                    time.sleep(0.05)
        finally:
            print(_banner("KARAOKÊ • STREAMS CLOSED"))
            self.publish("streams_closed")

        if self._win_event.is_set():
            return

        # computa resultado
        if frames_valid > 0:
            hit_ratio = float(frames_hits / frames_valid)
            achieved = float(np.median(pitches)) if len(pitches) else None
            rms_mean = float(np.mean(rms_samples)) if rms_samples else 0.0
            rms_median = float(np.median(rms_samples)) if rms_samples else 0.0
            jitter_cents = float(np.std(dev_cents_samples)) if len(dev_cents_samples) > 1 else None
            median_dev_cents = float(np.median(np.abs(dev_cents_samples))) if dev_cents_samples else None
            self._metrics: Dict[str, Any] = {}

            self._result = SessionResult(
                state=SessionState.finished,
                result="ganhou" if (
                        hit_ratio >= WIN_RATIO_MIN and
                        voiced_seconds >= MIN_VOICED_SECONDS and
                        max_streak >= MIN_CONTIG_STREAK_S and
                        (jitter_cents is not None and jitter_cents <= MAX_JITTER_CENTS) and
                        (median_dev_cents is not None and median_dev_cents <= MAX_MEDIAN_DEV_CENTS)
                ) else "perdeu",
                hit_ratio=hit_ratio,
                max_streak_s=float(max_streak),
                target_hz=float(TARGET_PITCH_HZ),
                tolerance_hz=float(TOLERANCE_HZ),
                achieved_pitch_hz=achieved,
                frames_valid=int(frames_valid),
                duration_s=float(self._duration_s),
                error=None
            )
            self._metrics = {
                "voiced_seconds": float(voiced_seconds),
                "rms_mean": rms_mean,
                "rms_median": rms_median,
                "jitter_cents": jitter_cents,
                "median_dev_cents": median_dev_cents,
            }

            # Envie no publish final também
            self.publish("stats",
                         rms_mean=rms_mean, rms_median=rms_median,
                         jitter_cents=jitter_cents, median_dev_cents=median_dev_cents,
                         voiced_seconds=float(voiced_seconds)
                         )
        else:
            # nenhum frame válido: gera resultado “perdeu” com erro descritivo
            self._result = SessionResult(
                state=SessionState.finished,
                result="perdeu",
                hit_ratio=0.0,
                max_streak_s=0.0,
                target_hz=float(TARGET_PITCH_HZ),
                tolerance_hz=float(TOLERANCE_HZ),
                achieved_pitch_hz=None,
                frames_valid=0,
                duration_s=float(self._duration_s),
                error=("librosa não instalado; sem medição"
                       if not HAS_LIBROSA else
                       "sem medições válidas (rms baixo/ruído/overflow ou sem voz detectada)")
            )
            self._metrics = {
                "voiced_seconds": 0.0,
                "rms_mean": 0.0,
                "rms_median": 0.0,
                "jitter_cents": None,
                "median_dev_cents": None,
            }

        self.state = SessionState.finished

    def _finalize_and_print(self) -> None:
        d = self.result_summary()
        print(_banner("🎤 RESULTADO DA SESSÃO KARAOKÊ"))
        print(_kv("status", d.get("result")))
        print(_kv("target_hz", f'{d.get("target_hz")} Hz'))
        print(_kv("achieved_hz", f'{d.get("achieved_pitch_hz")} Hz'))
        print(_kv("hit_ratio", f'{d.get("hit_ratio")}'))
        print(_kv("max_streak_s", f'{d.get("max_streak_s")}'))
        print(_kv("frames_valid", f'{d.get("frames_valid")}'))
        print(_kv("duration_s", f'{d.get("duration_s")}'))
        print(_kv("tolerance_hz", f'{d.get("tolerance_hz")}'))
        # se você adicionou ao summary, printe-os:
        print(_kv("voiced_seconds", f'{d.get("voiced_seconds", "—")}'))
        print(_kv("rms_mean", f'{d.get("rms_mean", "—")}'))
        print(_kv("rms_median", f'{d.get("rms_median", "—")}'))
        print(_kv("jitter_cents", f'{d.get("jitter_cents", "—")}'))
        print(_kv("median_dev_cents", f'{d.get("median_dev_cents", "—")}'))

        self.publish("finished", **d)

        print(_banner("JSON"))
        print(json.dumps(d, indent=4, ensure_ascii=False))

    def _finish_with_error(self, msg: str) -> None:
        self._result = SessionResult(
            state=SessionState.finished,
            result="perdeu",
            hit_ratio=0.0,
            max_streak_s=0.0,
            target_hz=float(TARGET_PITCH_HZ),
            tolerance_hz=float(TOLERANCE_HZ),
            achieved_pitch_hz=None,
            frames_valid=0,
            duration_s=float(self._duration_s),
            error=msg
        )
        self.state = SessionState.finished
        print(_err(f"ERROR: {msg}"))
        self.publish("finished", **self.result_summary())

    # ---------------------- saída pública -----------------------------
    def result_summary(self) -> Dict[str, Any]:
        if self._result is None:
            return {"state": self.state.value, "error": "Sem resultado"}
        d = asdict(self._result)

        d["state"] = self._result.state.value if isinstance(self._result.state, SessionState) else str(
            self._result.state)
        for k in ("target_hz", "tolerance_hz", "achieved_pitch_hz", "hit_ratio", "max_streak_s", "duration_s"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        if d.get("frames_valid") is not None:
            d["frames_valid"] = int(d["frames_valid"])

        # 👇 inclui métricas extras para o _finalize_and_print e para o /result
        if hasattr(self, "_metrics") and isinstance(self._metrics, dict):
            d.update(self._metrics)

        return d

