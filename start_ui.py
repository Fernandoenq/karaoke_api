# start_ui.py
import tkinter as tk
import threading
import time
import json
import queue

class StartUI:
    def __init__(self, event_queue: "queue.Queue[str]"):
        self.q = event_queue
        self.root = tk.Tk()
        self.root.title("Karaokê • Sessão /start (ao vivo)")
        self.root.geometry("520x320")
        self._closing = False

        pad = {"padx": 10, "pady": 6, "sticky": "w"}

        # --- Labels sem StringVar ---
        r = 0
        self.lbl_state  = self._row("Estado:", r, pad);        r += 1
        self.lbl_count  = self._row("Countdown:", r, pad);     r += 1
        self.lbl_f0     = self._row("f0 (Hz):", r, pad, big=True); r += 1
        self.lbl_cents  = self._row("Desvio (cents):", r, pad); r += 1
        self.lbl_ratio  = self._row("Hit ratio:", r, pad);     r += 1
        self.lbl_streak = self._row("Streak:", r, pad);        r += 1
        self.lbl_rms    = self._row("RMS:", r, pad);           r += 1

        self.text = tk.Text(self.root, height=6)
        self.text.grid(row=r, column=0, columnspan=2, padx=10, pady=6, sticky="nsew"); r += 1
        self.root.grid_rowconfigure(r-1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        btn = tk.Button(self.root, text="Fechar janela (não para a sessão)", command=self._safe_close)
        btn.grid(row=r, column=0, columnspan=2, pady=8)

        self.root.after(100, self._pump)

    def _row(self, title, r, pad, big=False):
        tk.Label(self.root, text=title).grid(row=r, column=0, **pad)
        lbl = tk.Label(self.root, text="—", font=("Calibri", 14, "bold") if big else None)
        lbl.grid(row=r, column=1, **pad)
        return lbl

    def _set(self, lbl: tk.Label, value):
        try:
            if self._closing or not self.root.winfo_exists():  # já fechando
                return
            lbl.config(text=str(value))
        except Exception:
            pass

    def _log(self, msg: str):
        try:
            if self._closing or not self.root.winfo_exists():
                return
            self.text.insert("end", msg + "\n")
            self.text.see("end")
        except Exception:
            pass

    def _safe_close(self):
        if self._closing:
            return
        self._closing = True
        # encerra loop do Tk no MESMO thread do Tk
        try:
            self.root.after(0, self.root.quit)
            self.root.after(50, self.root.destroy)
        except Exception:
            pass

    def _pump(self):
        if self._closing or not self.root.winfo_exists():
            return
        try:
            while True:
                s = self.q.get_nowait()
                try:
                    ev = json.loads(s)
                except Exception:
                    continue
                kind = ev.get("kind")

                if kind == "state":
                    self._set(self.lbl_state, ev.get("state", "—"))

                elif kind == "countdown":
                    self._set(self.lbl_count, ev.get("left", "—"))

                elif kind == "audio_resolved":
                    self._log(f'Áudio: {ev.get("file")} • {ev.get("duration_s")} s')

                elif kind == "pitch":
                    self._set(self.lbl_f0, f'{ev.get("f0", 0.0):.1f}')
                    self._set(self.lbl_ratio, f'{ev.get("ratio", 0.0):.2f}')
                    self._set(self.lbl_streak, f'{ev.get("streak_s", 0.0):.2f} s')
                    self._set(self.lbl_rms, f'{ev.get("rms", 0.0):.3f}')
                    dc = ev.get("dev_cents")
                    self._set(self.lbl_cents, "—" if dc is None else f'{dc:+.1f}')
                    # pinta leve conforme “hit”
                    try:
                        self.root.configure(bg="#eaffea" if ev.get("hit") else "#ffeaea")
                    except Exception:
                        pass

                elif kind == "win":
                    self._log(f'🏆 Vitória antecipada! ratio={ev.get("ratio"):.2f}')
                    self.root.after(300, self._safe_close)

                elif kind == "finished":
                    self._log("Sessão finalizada.")
                    self.root.after(300, self._safe_close)

                elif kind == "warn":
                    self._log(f'⚠️ {ev.get("message","")}')
                elif kind == "error":
                    self._log(f'❌ {ev.get("where","")}: {ev.get("message","")}')
        except queue.Empty:
            pass
        # repete o polling
        self.root.after(100, self._pump)


def open_start_ui(engine) -> None:
    """
    Registra um observer no engine e abre a janela em thread separada.
    Fechar a janela não interfere na sessão.
    """
    q = queue.Queue(maxsize=1000)
    engine.add_observer(q)

    def _run():
        try:
            ui = StartUI(q)
            ui.root.mainloop()
        finally:
            try:
                engine.remove_observer(q)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
