import time
import threading
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import aubio

# ====== CORES p/ logs bonitos ======
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    GREEN = Fore.GREEN; RED = Fore.RED; CYAN = Fore.CYAN; YELLOW = Fore.YELLOW; MAG = Fore.MAGENTA; RESET = Style.RESET_ALL; BOLD = Style.BRIGHT
except Exception:
    GREEN = RED = CYAN = YELLOW = MAG = RESET = BOLD = ""

BOX = "═" * 58
def box(title: str):
    print(f"\n{BOX}\n{BOLD}{title}{RESET}\n{BOX}")

def kv(key: str, val: str):  # formata coluna
    print(f"{key:<18}: {val}")

# ====== ESTADO ======
class SessionState(str, Enum):
    idle = "idle"
    countdown = "countdown"
    running = "running"
    finished = "finished"

@dataclass
class Result:
    state: SessionState
    result: Optional[str] = None           # "ganhou" | "perdeu"
    hit_ratio: Optional[float] = None
    max_streak_s: Optional[float] = None
    target_hz: Optional[float] = None
    tolerance_hz: Optional[float] = None
    achieved_pitch_hz: Optional[float] = None
    frames_valid: Optional[int] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None

# ====== DISPOSITIVOS ======
def list_audio_devices() -> Dict[str, Any]:
    """Retorna devices de áudio (entrada/saída) em formato amigável."""
    info = sd.query_devices()
    defaults = sd.default.device  # [in, out]
    inputs, outputs = [], []
    for idx, dev in enumerate(info):
        rec = dict(id=idx, name=dev["name"], max_input_channels=dev["max_input_channels"],
                   max_output_channels=dev["max_output_channels"])
        if dev["max_input_channels"] > 0:
            inputs.append(rec)
        if dev["max_output_channels"] > 0:
            outputs.append(rec)
    return {"defaults": defaults, "input": inputs, "output": outputs}

# ====== ENGINE ======
class GameEngine:
    def __init__(self):
        # parâmetros do jogo
        self.target_hz = 98.83           # alvo (pode vir do front depois)
        self.tolerance_hz = 40.0         # zona de acerto ±
        self.min_amp_threshold = 0.02    # rejeita ruído muito baixo
        self.buffer_size = 2048
        self.samplerate = 44100
        self.hop_size = self.buffer_size // 2

        # estado
        self.state: SessionState = SessionState.idle
        self._countdown_s = 3
        self._countdown_started_at: Optional[float] = None
        self._start_ts: Optional[float] = None
        self._stop_flag = threading.Event()

        # áudio/música
        self.music_path: Optional[Path] = None
        self.music_data: Optional[np.ndarray] = None
        self.music_sr: Optional[int] = None
        self.music_duration: Optional[float] = None

        # devices selecionados
        self.in_dev: Optional[int] = None
        self.out_dev: Optional[int] = None

        # medição
        self._pitch_o = aubio.pitch("yin", self.buffer_size, self.hop_size, self.samplerate)
        self._pitch_o.set_unit("Hz")
        self._pitch_o.set_silence(-90)
        self._alpha = 0.18  # suavização exponencial

        # métricas
        self._frames_valid = 0
        self._hit_frames = 0
        self._max_streak_s = 0.0
        self._current_streak_s = 0.0
        self._last_voice_pitch = 0.0
        self._achieved_pitch_avg = 0.0
        self._achieved_pitch_n = 0

        self._result: Optional[Result] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- helpers ----------
    def _resolve_music(self, music_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float], Optional[str]]:
        """Carrega o arquivo de música (.wav)."""
        # tentativa 1: arg explícito
        if music_path:
            p = Path(music_path).expanduser()
        else:
            # tentativa 2: arquivo padrão no projeto
            p = Path("referencia_trecho.wav")

        # se ainda não existir, tenta relativo ao diretório do projeto (root)
        if not p.exists():
            root = Path(__file__).resolve().parents[1]  # pasta do projeto
            p2 = (root / p).resolve()
            if p2.exists():
                p = p2

        # log de preparação
        box(f"{CYAN}KARAOKÊ • RESOLVE AUDIO{RESET}")
        msg = f"ok: {p}" if p.exists() else f"{RED}not found: {p}{RESET}"
        kv("resolve", msg)
        kv("cwd", str(Path('.').resolve()))

        if not p.exists():
            return None, None, None, f"Arquivo não encontrado: {p}"

        try:
            data, sr = sf.read(str(p), dtype='float32', always_2d=True)
            mono = data[:, 0]  # usa canal L
            duration = len(mono) / sr
            self.music_path = p
            self.music_data = mono
            self.music_sr = sr
            self.music_duration = duration

            box(f"{GREEN}KARAOKÊ • AUDIO OK{RESET}")
            kv("file", p.name)
            kv("duration", f"{duration:.2f}s")
            return mono, sr, duration, None
        except Exception as e:
            return None, None, None, f"Erro abrindo WAV: {e}"

    def prepare(self, music_path: Optional[str], in_dev: Optional[int], out_dev: Optional[int]) -> None:
        """Reseta estado e entra em countdown."""
        self.stop()  # garante que não há sessão antiga
        self._stop_flag.clear()
        self._result = None

        # devices
        self.in_dev = in_dev
        self.out_dev = out_dev

        # logs bonitos
        box(f"{MAG}KARAOKÊ • PREPARE{RESET}")
        kv("input dev", str(self.in_dev if self.in_dev is not None else "default"))
        kv("output dev", str(self.out_dev if self.out_dev is not None else "default"))
        kv("music_path", str(music_path) if music_path else "referencia_trecho.wav")
        kv("cwd", str(Path('.').resolve()))
        kv("project_root", str(Path(__file__).resolve().parents[1]))

        # carrega música aqui? não — deixamos para a thread (com logs)
        self.state = SessionState.countdown
        self._countdown_started_at = time.time()

    def countdown_left(self) -> Optional[int]:
        if self.state != SessionState.countdown or self._countdown_started_at is None:
            return None
        left = self._countdown_s - int(time.time() - self._countdown_started_at)
        return max(0, left)

    def progress(self) -> float:
        if self.state != SessionState.running or self._start_ts is None or not self.music_duration:
            return 0.0
        elapsed = time.time() - self._start_ts
        return float(np.clip(elapsed / self.music_duration, 0.0, 1.0))

    def status_message(self) -> str:
        if self.state == SessionState.idle:
            return "Aguardando início"
        if self.state == SessionState.countdown:
            return f"Contagem: {self.countdown_left()}s"
        if self.state == SessionState.running:
            return "Tocando e medindo…"
        if self.state == SessionState.finished:
            return "Sessão finalizada"
        return ""

    def result_summary(self) -> Dict[str, Any]:
        if self._result is None:
            return {"state": self.state, "error": "Sem resultado"}
        return asdict(self._result)

    def stop(self) -> None:
        """Sinaliza cancelamento; usada também no prepare para reset limpo."""
        if self.state in (SessionState.countdown, SessionState.running):
            print(f"{YELLOW}[KARAOKÊ] stop requested{RESET}")
        self._stop_flag.set()

    # ---------- main async ----------
    def run_session_async(self):
        """
        Roda o ciclo completo em thread:
        countdown → carregar música → tocar/analisar → fechar → finalizar.
        """
        # espera countdown
        while self.state == SessionState.countdown and not self._stop_flag.is_set():
            if self.countdown_left() == 0:
                break
            time.sleep(0.2)

        if self._stop_flag.is_set():
            self.state = SessionState.idle
            return

        # carrega música
        data, sr, dur, err = self._resolve_music(None if self.music_path is None else str(self.music_path))
        if data is None:
            # se não veio do prepare, tenta novamente com "referencia_trecho.wav"
            data, sr, dur, err = self._resolve_music("referencia_trecho.wav")
        if data is None:
            # falha definitiva
            print(f"{RED}[KARAOKÊ] ERROR loading audio: {err}{RESET}")
            self._result = Result(
                state=SessionState.finished,
                result=None,
                error=str(err)
            )
            self.state = SessionState.finished
            return

        # toca e mede
        self.state = SessionState.running
        self._start_ts = time.time()
        self._run_play_and_measure(data, sr)

        # monta resultado
        hit_ratio = (self._hit_frames / max(1, self._frames_valid))
        achieved = (self._achieved_pitch_avg / max(1, self._achieved_pitch_n)) if self._achieved_pitch_n else None
        verdict = "ganhou" if hit_ratio >= 0.5 else "perdeu"  # regra simples — ajuste depois

        self._result = Result(
            state=SessionState.finished,
            result=verdict,
            hit_ratio=round(hit_ratio, 3),
            max_streak_s=round(self._max_streak_s, 2),
            target_hz=round(self.target_hz, 2),
            tolerance_hz=self.tolerance_hz,
            achieved_pitch_hz=(round(achieved, 2) if achieved else None),
            frames_valid=self._frames_valid,
            duration_s=self.music_duration,
            error=None
        )

        # log final (apenas ao terminar)
        box(f"{BOLD}🎤 RESULTADO DA SESSÃO KARAOKÊ{RESET}")
        kv("status", f"{GREEN}{verdict}{RESET}" if verdict == "ganhou" else f"{RED}{verdict}{RESET}")
        kv("target_hz", f"{self.target_hz:.2f} Hz")
        kv("achieved_hz", f"{achieved:.2f} Hz" if achieved else "—")
        kv("hit_ratio", f"{hit_ratio:.3f}")
        kv("max_streak_s", f"{self._max_streak_s:.2f}")
        kv("frames_valid", f"{self._frames_valid}")
        kv("duration_s", f"{self.music_duration:.1f}")
        kv("tolerance_hz", f"{self.tolerance_hz:.1f}")

        # pretty JSON para copiar/colar
        import json
        box("JSON")
        print(json.dumps(asdict(self._result), indent=4, ensure_ascii=False))

        self.state = SessionState.finished

    # ---------- audio run ----------
    def _run_play_and_measure(self, music: np.ndarray, sr: int):
        """
        Abre duas streams:
        - OutputStream: toca a música (buffer por buffer)
        - InputStream : lê microfone, calcula pitch e atualiza métricas
        """
        box(f"{CYAN}KARAOKÊ • OPEN STREAMS{RESET}")
        kv("in dev", str(self.in_dev if self.in_dev is not None else "default"))
        kv("out dev", str(self.out_dev if self.out_dev is not None else "default"))
        kv("default", str(sd.default.device))

        # prepara índices e flags
        idx = 0
        total_frames = len(music)
        frames_per_block = self.hop_size

        # callbacks
        def out_cb(outdata, frames, time_info, status):
            nonlocal idx
            if status:
                print(f"{YELLOW}[OUTPUT] {status}{RESET}")
            end = min(idx + frames, total_frames)
            chunk = music[idx:end]
            if len(chunk) < frames:
                # preenche zeros no final
                pad = np.zeros(frames - len(chunk), dtype=music.dtype)
                chunk = np.concatenate([chunk, pad])
            outdata[:, 0] = chunk
            idx = end
            if idx >= total_frames:
                raise sd.CallbackStop()

        def in_cb(indata, frames, time_info, status):
            if status:
                print(f"{YELLOW}[INPUT] {status}{RESET}")
            samples = indata[:, 0].astype(np.float32)
            pitch = self._pitch_o(samples)[0]
            amp = np.max(np.abs(samples))
            if 50.0 < pitch < 2000.0 and amp > self.min_amp_threshold:
                # suavização
                self._last_voice_pitch = (1 - self._alpha) * self._last_voice_pitch + self._alpha * pitch
                self._frames_valid += 1
                # métricas
                if abs(self._last_voice_pitch - self.target_hz) <= self.tolerance_hz:
                    self._hit_frames += 1
                    self._current_streak_s += frames / self.samplerate
                    self._max_streak_s = max(self._max_streak_s, self._current_streak_s)
                else:
                    self._current_streak_s = 0.0
                # acumulador de média
                self._achieved_pitch_avg += self._last_voice_pitch
                self._achieved_pitch_n += 1

        # toca e mede
        box(f"{GREEN}KARAOKÊ • PLAYING & MEASURING{RESET}")
        try:
            with sd.InputStream(
                device=self.in_dev,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.hop_size,
                callback=in_cb,
            ), sd.OutputStream(
                device=self.out_dev,
                channels=1,
                samplerate=sr,
                blocksize=self.hop_size,
                callback=out_cb,
            ):
                while idx < total_frames and not self._stop_flag.is_set():
                    sd.sleep(20)
        except Exception as e:
            # falha ao abrir stream — guarda como resultado final também
            msg = str(e)
            print(f"{RED}[KARAOKÊ] PortAudio ERROR: {msg}{RESET}")
            self._result = Result(
                state=SessionState.finished,
                result=None,
                error=f"PortAudio: {msg}"
            )
        finally:
            box(f"{YELLOW}KARAOKÊ • STREAMS CLOSED{RESET}")
