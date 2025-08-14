import os
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init as colorama_init

# Tente importar aubio para pitch; se não houver, segue sem medir (resultado "perdeu")
try:
    import aubio  # type: ignore
    HAS_AUBIO = True
except Exception:
    HAS_AUBIO = False

colorama_init(autoreset=True)

# ======================= CONFIG PADRÃO DO JOGO =======================
TARGET_PITCH_HZ = 440        # ~ G2
TOLERANCE_HZ    = 40.0
COUNTDOWN_S     = 3            # 3,2,1 e começa

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

# ======================= ENGINE =====================================
class GameEngine:
    def __init__(self):
        self.state: SessionState = SessionState.idle
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        self._play_t: int = 0  # <-- ponteiro de reprodução (frames), AGORA na instância

        # resultado
        self._result: Optional[SessionResult] = None

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
        self._result = None
        self._music_path = music_path or "referencia_trecho.wav"  # fallback (relativo ao cwd)
        self._in_dev = in_dev
        self._out_dev = out_dev
        self._play_t = 0  # zera o ponteiro de reprodução
        self.state = SessionState.countdown

        print(_banner("KARAOKÊ • PREPARE"))
        print(_kv("input dev", self._in_dev))
        print(_kv("output dev", self._out_dev))
        print(_kv("music_path", self._music_path))
        print(_kv("cwd", os.getcwd()))
        print(_kv("project_root", os.path.abspath(".")))

    def stop(self) -> None:
        self._stop_flag.set()
        self.state = SessionState.stopping
        print(_warn("stop requested"))

    def run_session_async(self) -> None:
        try:
            # -------- countdown ----------
            for i in range(COUNTDOWN_S, 0, -1):
                if self._stop_flag.is_set():
                    self.state = SessionState.finished
                    return
                print(_kv("countdown", i, Fore.CYAN))
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

            # -------- abrir streams ----------
            print(_banner("KARAOKÊ • OPEN STREAMS"))
            print(_kv("in dev", self._in_dev))
            print(_kv("out dev", self._out_dev))
            print(_kv("default", str(sd.default.device)))

            self.state = SessionState.running
            self._t_start = time.time()

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

    def _play_and_measure(self, path: str) -> None:
        if self._audio_data is None or self._sr is None:
            raise RuntimeError("Áudio não carregado.")

        data = self._audio_data
        sr = self._sr

        # configurando pitch detect se houver aubio
        if HAS_AUBIO:
            win_s = 1024
            hop_s = 512
            pitch_o = aubio.pitch("yin", win_s, hop_s, sr)
            pitch_o.set_unit("Hz")
            pitch_o.set_silence(-40)

            frames_hits = 0
            frames_valid = 0
            max_streak = 0
            current_streak = 0
            pitches: List[float] = []

        # callback do output: só toca a música
        def out_cb(outdata, frames, time_info, status):
            del time_info
            if status:
                print(_warn(f"OUTPUT status: {status}"))
            # pegar o slice correspondente usando o ponteiro da INSTÂNCIA
            t = self._play_t
            end = t + frames
            if end > len(data):
                end = len(data)
            chunk = data[t:end]
            if len(chunk) < frames:
                pad = np.zeros((frames - len(chunk), data.shape[1]), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            outdata[:] = chunk
            self._play_t = end  # avança o ponteiro

        # callback do input: mede pitch (se aubio disponível)
        def in_cb(indata, frames, time_info, status):
            del time_info
            if status:
                print(_warn(f"INPUT status: {status}"))
            if not HAS_AUBIO:
                return

            nonlocal frames_hits, frames_valid, max_streak, current_streak, pitches  # noqa: F821

            mono = np.mean(indata, axis=1).astype(np.float32)
            hop = 512
            for i in range(0, len(mono), hop):
                buf = mono[i:i+hop]
                if len(buf) < hop:
                    pad = np.zeros(hop - len(buf), dtype=np.float32)
                    buf = np.concatenate([buf, pad])
                p = float(pitch_o(buf)[0])
                if np.isfinite(p) and p > 0:
                    frames_valid += 1
                    pitches.append(p)
                    if abs(p - TARGET_PITCH_HZ) <= TOLERANCE_HZ:
                        frames_hits += 1
                        current_streak += hop / sr
                        if current_streak > max_streak:
                            max_streak = current_streak
                    else:
                        current_streak = 0

        # abrir streams
        try:
            with sd.OutputStream(
                samplerate=sr,
                channels=data.shape[1],
                device=self._out_dev,
                callback=out_cb,
                dtype="float32",
                finished_callback=lambda: None,
            ), sd.InputStream(
                samplerate=sr,
                channels=1,
                device=self._in_dev,
                callback=in_cb,
                dtype="float32",
            ):
                # roda até acabar a música ou pedirem stop
                while not self._stop_flag.is_set() and self._play_t < len(data):
                    time.sleep(0.05)
        finally:
            print(_banner("KARAOKÊ • STREAMS CLOSED"))

        # computa resultado
        if HAS_AUBIO:
            hit_ratio = float(frames_hits / frames_valid) if frames_valid else 0.0
            achieved = float(np.median(pitches)) if len(pitches) else None
            self._result = SessionResult(
                state=SessionState.finished,
                result="ganhou" if hit_ratio >= 0.6 else "perdeu",
                hit_ratio=hit_ratio,
                max_streak_s=float(max_streak),
                target_hz=float(TARGET_PITCH_HZ),
                tolerance_hz=float(TOLERANCE_HZ),
                achieved_pitch_hz=achieved,
                frames_valid=int(frames_valid),
                duration_s=float(self._duration_s),
                error=None
            )
        else:
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
                error="aubio não instalado; sem medição"
            )

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

        print(_banner("JSON"))
        import json
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

    # ---------------------- saída pública -----------------------------
    def result_summary(self) -> Dict[str, Any]:
        if self._result is None:
            return {"state": self.state.value, "error": "Sem resultado"}
        d = asdict(self._result)

        # garantir tipos nativos & state como string
        d["state"] = self._result.state.value if isinstance(self._result.state, SessionState) else str(self._result.state)
        for k in ("target_hz", "tolerance_hz", "achieved_pitch_hz", "hit_ratio", "max_streak_s", "duration_s"):
            if d.get(k) is not None:
                d[k] = float(d[k])
        if d.get("frames_valid") is not None:
            d["frames_valid"] = int(d["frames_valid"])
        return d
