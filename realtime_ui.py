# realtime_ui.py
import threading
import time
import queue
from typing import Optional, List, Any, Dict

import numpy as np
import sounddevice as sd

# Tkinter para UI simples
import tkinter as tk

# YIN (librosa)
try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# ===== Opcional: reaproveita seus "knobs" do game_logic =====
# Se preferir, importe direto de game_logic para ficar 100% consistente:
#   from .game_logic import (FMIN_HZ, FMAX_HZ, TARGET_PITCH_HZ, TOLERANCE_HZ,
#                            MIN_RMS, MIN_REQUIRED_PITCH_HZ, ANY_HIT_MODE, ANY_HIT_MIN_STREAK_S)
# Para deixar este módulo independente, mantenho defaults abaixo:
FMIN_HZ = 90.0
FMAX_HZ = 900.0
TARGET_PITCH_HZ = 330.0
TOLERANCE_HZ = 40.0
MIN_RMS = 0.02
MIN_REQUIRED_PITCH_HZ = 180.0
ANY_HIT_MODE = True
ANY_HIT_MIN_STREAK_S = 0.10

CENTS_TOL = 30.0  # tolerância de acerto em cents

# ======================= HELPERS =======================
def _rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))

def _hz_to_cents_ratio(f_hz: float, ref_hz: float) -> float:
    if f_hz <= 0 or ref_hz <= 0:
        return float("inf")
    return 1200.0 * np.log2(f_hz / ref_hz)

def _nearest_octave_ref(f_hz: float, target_hz: float) -> float:
    if f_hz <= 0 or target_hz <= 0:
        return target_hz
    k = round(np.log2(f_hz / target_hz))
    return target_hz * (2.0 ** k)

def _is_hit_strict_cents(f_hz: float, target_hz: float, cents_tol: float) -> bool:
    if f_hz <= 0:
        return False
    ref = _nearest_octave_ref(f_hz, target_hz)
    dev_cents = abs(_hz_to_cents_ratio(f_hz, ref))
    return dev_cents <= cents_tol

def _is_hit(pitch_hz: float, target_hz: float, tol_hz: float) -> bool:
    # usa cents por precisão e evitar oitava errada
    return _is_hit_strict_cents(pitch_hz, target_hz, cents_tol=CENTS_TOL)

class LibrosaPitchDetector:
    def __init__(self, sr: int, fmin: float = FMIN_HZ, fmax: float = FMAX_HZ,
                 frame_length: int = 2048, hop_length: int = 2048,
                 periodicity_threshold: float = 0.08):
        self.sr = int(sr)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.periodicity_threshold = float(periodicity_threshold)
        self._buffer = np.zeros(0, dtype=np.float32)

    def process_block(self, mono_block: np.ndarray) -> List[float]:
        if mono_block.ndim != 1:
            mono_block = np.mean(mono_block, axis=-1)
        mono_block = mono_block.astype(np.float32, copy=False)

        self._buffer = np.concatenate([self._buffer, mono_block])
        f0_list: List[float] = []

        while len(self._buffer) >= self.frame_length:
            frame = self._buffer[-self.frame_length:]
            try:
                f0 = librosa.yin(
                    y=frame,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    frame_length=self.frame_length,
                    hop_length=self.frame_length,
                    trough_threshold=self.periodicity_threshold,
                )
                hz = float(f0[0]) if np.isfinite(f0[0]) else 0.0
            except Exception:
                hz = 0.0

            f0_list.append(hz if (hz > 0.0 and np.isfinite(hz)) else 0.0)
            self._buffer = self._buffer[self.hop_length:]

        return f0_list

# ======================= UI ENGINE =======================
class RealtimeMonitor:
    """
    Captura microfone e publica métricas em uma UI simples (Tkinter).
    """
    def __init__(self, in_dev: Optional[int], sr: int = 44100, blocksize: int = 2048):
        self.in_dev = in_dev
        self.sr = int(sr)
        self.blocksize = int(blocksize)

        self._stop = threading.Event()
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)

        self.frames_hits = 0
        self.frames_valid = 0
        self.current_streak = 0.0
        self.max_streak = 0.0

        self.detector = None
        if HAS_LIBROSA:
            self.detector = LibrosaPitchDetector(
                sr=self.sr,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                frame_length=self.blocksize,
                hop_length=self.blocksize,
                periodicity_threshold=0.08,
            )

        # UI
        self.root = tk.Tk()
        self.root.title("Análise de Voz • Tempo Real")
        self.root.geometry("420x260")

        # labels
        self.var_f0      = tk.StringVar(value="—")
        self.var_hit     = tk.StringVar(value="—")
        self.var_ratio   = tk.StringVar(value="0.00")
        self.var_streak  = tk.StringVar(value="0.00 s")
        self.var_rms     = tk.StringVar(value="0.000")
        self.var_cents   = tk.StringVar(value="—")

        pad = {"padx": 10, "pady": 6, "sticky": "w"}
        tk.Label(self.root, text="f0 (Hz):").grid(row=0, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_f0, font=("Calibri", 14, "bold")).grid(row=0, column=1, **pad)

        tk.Label(self.root, text="Hit (±cents):").grid(row=1, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_hit, font=("Calibri", 12)).grid(row=1, column=1, **pad)

        tk.Label(self.root, text="Desvio (cents):").grid(row=2, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_cents).grid(row=2, column=1, **pad)

        tk.Label(self.root, text="Hit ratio:").grid(row=3, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_ratio).grid(row=3, column=1, **pad)

        tk.Label(self.root, text="Streak:").grid(row=4, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_streak).grid(row=4, column=1, **pad)

        tk.Label(self.root, text="RMS:").grid(row=5, column=0, **pad)
        tk.Label(self.root, textvariable=self.var_rms).grid(row=5, column=1, **pad)

        btn = tk.Button(self.root, text="Encerrar", command=self.close)
        btn.grid(row=6, column=0, columnspan=2, pady=12)

        # fecha limpa
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self._stop.set()
        try:
            self.stream.abort(ignore_errors=True)  # type: ignore
            self.stream.close()                    # type: ignore
        except Exception:
            pass
        self.root.destroy()

    def _update_ui_loop(self):
        if self._stop.is_set():
            return
        try:
            while True:
                d = self._q.get_nowait()
                # atualiza labels
                self.var_f0.set(f'{d.get("f0", 0.0):.1f}')
                self.var_ratio.set(f'{d.get("ratio", 0.0):.2f}')
                self.var_streak.set(f'{d.get("streak_s", 0.0):.2f} s')
                self.var_rms.set(f'{d.get("rms", 0.0):.3f}')
                cents = d.get("dev_cents")
                if cents is None:
                    self.var_cents.set("—")
                else:
                    self.var_cents.set(f'{cents:+.1f}')
                hit = d.get("hit", False)
                color = "green" if hit else "red"
                self.var_hit.set("ACERTOU" if hit else "—")
                # pinta a janela levemente
                self.root.configure(bg="#eaffea" if hit else "#ffeaea")
        except queue.Empty:
            pass
        # agenda próximo tick
        self.root.after(100, self._update_ui_loop)

    def _audio_thread(self):
        if not HAS_LIBROSA:
            # publica um aviso na UI
            self._q.put({"f0": 0.0, "ratio": 0.0, "streak_s": 0.0, "rms": 0.0, "dev_cents": None, "hit": False})
            return

        def in_cb(indata, frames, time_info, status):
            del time_info
            if status:
                # você pode logar se quiser
                pass

            mono = np.mean(indata, axis=1).astype(np.float32)
            block_rms = _rms(np.asarray(indata, dtype=np.float32).reshape(-1))
            f0_list = self.detector.process_block(mono) if self.detector else []

            for f0 in f0_list:
                is_pitch_ok = (f0 > 0.0 and np.isfinite(f0))
                is_loud     = block_rms >= MIN_RMS
                is_high     = (f0 >= MIN_REQUIRED_PITCH_HZ) if is_pitch_ok else False

                if is_pitch_ok and is_loud and is_high:
                    self.frames_valid += 1
                    ref = _nearest_octave_ref(f0, TARGET_PITCH_HZ)
                    dev_cents = _hz_to_cents_ratio(f0, ref)
                    hit = _is_hit(f0, TARGET_PITCH_HZ, TOLERANCE_HZ)

                    if hit:
                        self.frames_hits += 1
                        self.current_streak += frames / self.sr
                        if self.current_streak > self.max_streak:
                            self.max_streak = self.current_streak
                    else:
                        self.current_streak = 0.0

                    ratio = self.frames_hits / max(1, self.frames_valid)
                    try:
                        self._q.put_nowait({
                            "f0": float(f0),
                            "hit": bool(hit),
                            "ratio": float(ratio),
                            "streak_s": float(self.current_streak),
                            "rms": float(block_rms),
                            "dev_cents": float(dev_cents),
                        })
                    except queue.Full:
                        pass
                else:
                    self.current_streak = 0.0

        # abre stream de entrada
        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            device=self.in_dev,
            callback=in_cb,
            dtype="float32",
            blocksize=self.blocksize,
            latency="high",
        ) as self.stream:
            while not self._stop.is_set():
                time.sleep(0.05)

    def run(self):
        # inicia o “poll” da UI
        self.root.after(100, self._update_ui_loop)
        # inicia áudio em thread separada
        t = threading.Thread(target=self._audio_thread, daemon=True)
        t.start()
        # loop da UI (bloqueia até fechar janela)
        self.root.mainloop()
        self._stop.set()

def launch_realtime_ui(in_dev: Optional[int], sr: int = 44100, blocksize: int = 2048):
    """
    Função simples para ser chamada pela rota /teste.
    """
    monitor = RealtimeMonitor(in_dev=in_dev, sr=sr, blocksize=blocksize)
    monitor.run()
