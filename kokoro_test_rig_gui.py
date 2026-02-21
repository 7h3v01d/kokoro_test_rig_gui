"""
Kokoro-82M-v1.0-ONNX Test Rig (GUI)
+ Waveform preview
+ Spectrogram preview
+ RMS / Peak loudness meter (dBFS)
+ Cursor readout (time/amplitude or time/frequency)

Expected files (default, next to this script):
  - kokoro-v1.0.onnx
  - voices-v1.0.bin

Install:
  pip install -U kokoro-onnx numpy matplotlib
  pip install -U sounddevice   # optional
"""

from __future__ import annotations

import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from kokoro_onnx import Kokoro
except Exception as e:
    raise SystemExit(
        "Failed to import kokoro_onnx. Install with:\n"
        "  pip install -U kokoro-onnx\n\n"
        f"Import error: {e}"
    )

# sounddevice is optional (only needed for speaker playback)
try:
    import sounddevice as sd  # type: ignore

    _HAS_SD = True
except Exception:
    sd = None
    _HAS_SD = False

# matplotlib for preview panels
try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except Exception as e:
    raise SystemExit(
        "Failed to import matplotlib Tk backend. Install with:\n"
        "  pip install matplotlib\n\n"
        f"Import error: {e}"
    )


DEFAULT_VOICES = [
    "af_sarah",
    "af_bella",
    "af_nicole",
    "af_sky",
    "af_river",
    "am_adam",
    "am_michael",
    "am_onyx",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_george",
    "bm_lewis",
]


@dataclass
class GeneratedAudio:
    samples: np.ndarray  # float32 waveform [-1..1]
    sample_rate: int
    meta: str
    rms_dbfs: float
    peak_dbfs: float


def _write_wav_pcm16(path: str, samples_f32: np.ndarray, sample_rate: int) -> None:
    """Save float waveform [-1..1] to 16-bit PCM WAV (mono)."""
    if samples_f32.ndim != 1:
        samples_f32 = np.asarray(samples_f32).reshape(-1)

    s = np.clip(samples_f32, -1.0, 1.0)
    pcm16 = (s * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _dbfs_from_rms(rms: float) -> float:
    # Avoid log(0)
    rms = float(max(rms, 1e-12))
    return 20.0 * np.log10(rms)


def _dbfs_from_peak(peak: float) -> float:
    peak = float(max(peak, 1e-12))
    return 20.0 * np.log10(peak)


def _compute_levels_dbfs(samples: np.ndarray) -> tuple[float, float]:
    s = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(s * s))) if s.size else 0.0
    peak = float(np.max(np.abs(s))) if s.size else 0.0
    return _dbfs_from_rms(rms), _dbfs_from_peak(peak)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class KokoroTestRig(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Kokoro-82M v1.0 ONNX — Test Rig")
        self.geometry("1220x780")
        self.minsize(1100, 700)

        self._kokoro: Optional[Kokoro] = None
        self._kokoro_paths: Optional[Tuple[str, str]] = None  # (model, voices)
        self._last_audio: Optional[GeneratedAudio] = None
        self._worker: Optional[threading.Thread] = None

        # Plot performance: decimate long waveforms for interactive redraw
        self._plot_max_points = 200_000

        # Waveform selection (seconds)
        self._selection_start_var = tk.DoubleVar(value=0.0)
        self._selection_end_var = tk.DoubleVar(value=2.0)

        # Spectrogram params
        self._spec_nfft_var = tk.IntVar(value=1024)
        self._spec_noverlap_var = tk.IntVar(value=768)

        # Cursor readout
        self.cursor_var = tk.StringVar(value="Cursor: —")

        # Loudness / meter
        self.rms_var = tk.StringVar(value="RMS: — dBFS")
        self.peak_var = tk.StringVar(value="Peak: — dBFS")
        self.meter_var = tk.IntVar(value=0)  # 0..100

        self._build_ui()
        self._set_defaults()

        # Matplotlib event bindings (cursor readout)
        self._bind_matplotlib_cursor_events()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(12, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Kokoro-82M-v1.0 (ONNX) — Local TTS Test Rig",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(header, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

        cfg = ttk.LabelFrame(self, text="Model + Synthesis Settings", padding=(12, 10))
        cfg.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
        cfg.columnconfigure(1, weight=1)
        cfg.columnconfigure(4, weight=1)

        self.model_path_var = tk.StringVar()
        ttk.Label(cfg, text="Model (.onnx):").grid(row=0, column=0, sticky="w")
        ttk.Entry(cfg, textvariable=self.model_path_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(cfg, text="Browse…", command=self._browse_model).grid(row=0, column=2, sticky="w")

        self.voices_path_var = tk.StringVar()
        ttk.Label(cfg, text="Voices (.bin):").grid(row=0, column=3, sticky="w")
        ttk.Entry(cfg, textvariable=self.voices_path_var).grid(row=0, column=4, sticky="ew", padx=(8, 8))
        ttk.Button(cfg, text="Browse…", command=self._browse_voices).grid(row=0, column=5, sticky="w")

        self.voice_var = tk.StringVar()
        ttk.Label(cfg, text="Voice:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.voice_combo = ttk.Combobox(cfg, textvariable=self.voice_var, values=DEFAULT_VOICES)
        self.voice_combo.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))

        self.lang_var = tk.StringVar()
        ttk.Label(cfg, text="Lang:").grid(row=1, column=3, sticky="w", pady=(10, 0))
        self.lang_combo = ttk.Combobox(cfg, textvariable=self.lang_var, values=["en-us", "en-gb"])
        self.lang_combo.grid(row=1, column=4, sticky="ew", padx=(8, 8), pady=(10, 0))

        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Label(cfg, text="Speed:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        speed_row = ttk.Frame(cfg)
        speed_row.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(10, 0))
        speed_row.columnconfigure(0, weight=1)
        self.speed_scale = ttk.Scale(speed_row, from_=0.5, to=2.0, variable=self.speed_var, orient="horizontal")
        self.speed_scale.grid(row=0, column=0, sticky="ew")
        self.speed_readout = ttk.Label(speed_row, text="1.00")
        self.speed_readout.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.speed_scale.bind("<Motion>", lambda _e: self._update_speed_readout())
        self.speed_scale.bind("<ButtonRelease-1>", lambda _e: self._update_speed_readout())

        self.output_mode = tk.StringVar(value="play" if _HAS_SD else "wav")
        out = ttk.Frame(cfg)
        out.grid(row=2, column=3, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(out, text="Output:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(out, text="Play to speakers", value="play", variable=self.output_mode).grid(
            row=0, column=1, sticky="w", padx=(10, 0)
        )
        ttk.Radiobutton(out, text="Save WAV", value="wav", variable=self.output_mode).grid(
            row=0, column=2, sticky="w", padx=(10, 0)
        )
        if not _HAS_SD:
            ttk.Label(out, text="(sounddevice not installed → playback disabled)", foreground="#888").grid(
                row=0, column=3, sticky="w", padx=(10, 0)
            )

        # Body split: left (text + actions), right (plots)
        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        text_frame = ttk.LabelFrame(left, text="Input Text", padding=(10, 8))
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.text = tk.Text(text_frame, wrap="word", height=12)
        self.text.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(left, text="Actions + Levels", padding=(10, 8))
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)

        self.btn_generate = ttk.Button(controls, text="Generate", command=self._generate_clicked)
        self.btn_generate.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.btn_play = ttk.Button(controls, text="Play Last", command=self._play_last)
        self.btn_play.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        self.btn_save = ttk.Button(controls, text="Save Last WAV…", command=self._save_last)
        self.btn_save.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.btn_stop = ttk.Button(controls, text="Stop Playback", command=self._stop_playback)
        self.btn_stop.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Separator(controls).grid(row=4, column=0, sticky="ew", pady=10)

        # Levels + meter
        lvl = ttk.Frame(controls)
        lvl.grid(row=5, column=0, sticky="ew")
        lvl.columnconfigure(0, weight=1)

        ttk.Label(lvl, textvariable=self.rms_var).grid(row=0, column=0, sticky="w")
        ttk.Label(lvl, textvariable=self.peak_var).grid(row=1, column=0, sticky="w", pady=(2, 0))

        meter_row = ttk.Frame(controls)
        meter_row.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        meter_row.columnconfigure(0, weight=1)
        ttk.Label(meter_row, text="Level (RMS):").grid(row=0, column=0, sticky="w")
        self.meter = ttk.Progressbar(meter_row, variable=self.meter_var, maximum=100)
        self.meter.grid(row=1, column=0, sticky="ew", pady=(3, 0))

        ttk.Separator(controls).grid(row=7, column=0, sticky="ew", pady=10)

        self.last_meta_var = tk.StringVar(value="No audio generated yet.")
        ttk.Label(controls, text="Last:", font=("Segoe UI", 9, "bold")).grid(row=8, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.last_meta_var, wraplength=360).grid(row=9, column=0, sticky="w")

        ttk.Button(controls, text="Copy Last Meta", command=self._copy_last_meta).grid(
            row=10, column=0, sticky="ew", pady=(10, 0)
        )

        # Right: Notebook (Waveform / Spectrogram)
        right = ttk.LabelFrame(body, text="Preview", padding=(10, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.grid(row=0, column=0, sticky="nsew")

        # --- Waveform tab
        self.tab_wave = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_wave.columnconfigure(0, weight=1)
        self.tab_wave.rowconfigure(2, weight=1)
        self.nb.add(self.tab_wave, text="Waveform")

        sel = ttk.Frame(self.tab_wave)
        sel.grid(row=0, column=0, sticky="ew")
        sel.columnconfigure(8, weight=1)

        ttk.Label(sel, text="View range (seconds):").grid(row=0, column=0, sticky="w")
        ttk.Label(sel, text="Start").grid(row=0, column=1, sticky="w", padx=(10, 2))
        ttk.Entry(sel, textvariable=self._selection_start_var, width=8).grid(row=0, column=2, sticky="w")
        ttk.Label(sel, text="End").grid(row=0, column=3, sticky="w", padx=(10, 2))
        ttk.Entry(sel, textvariable=self._selection_end_var, width=8).grid(row=0, column=4, sticky="w")
        ttk.Button(sel, text="Zoom", command=self._zoom_to_selection).grid(row=0, column=5, sticky="w", padx=(10, 0))
        ttk.Button(sel, text="Reset", command=self._reset_view).grid(row=0, column=6, sticky="w", padx=(6, 0))
        ttk.Button(sel, text="Refresh", command=self._update_waveform_plot).grid(row=0, column=7, sticky="w", padx=(6, 0))

        self.fig_wave = Figure(figsize=(6, 3), dpi=100)
        self.ax_wave = self.fig_wave.add_subplot(111)
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.2)

        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.tab_wave)
        self.canvas_wave.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        toolbar_wave = NavigationToolbar2Tk(self.canvas_wave, self.tab_wave, pack_toolbar=False)
        toolbar_wave.update()
        toolbar_wave.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        # --- Spectrogram tab
        self.tab_spec = ttk.Frame(self.nb, padding=(8, 8))
        self.tab_spec.columnconfigure(0, weight=1)
        self.tab_spec.rowconfigure(2, weight=1)
        self.nb.add(self.tab_spec, text="Spectrogram")

        spec_ctrl = ttk.Frame(self.tab_spec)
        spec_ctrl.grid(row=0, column=0, sticky="ew")
        spec_ctrl.columnconfigure(8, weight=1)

        ttk.Label(spec_ctrl, text="NFFT").grid(row=0, column=0, sticky="w")
        ttk.Entry(spec_ctrl, textvariable=self._spec_nfft_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 14))
        ttk.Label(spec_ctrl, text="Overlap").grid(row=0, column=2, sticky="w")
        ttk.Entry(spec_ctrl, textvariable=self._spec_noverlap_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 14))
        ttk.Button(spec_ctrl, text="Render", command=self._update_spectrogram_plot).grid(row=0, column=4, sticky="w")
        ttk.Button(spec_ctrl, text="Auto params", command=self._auto_spec_params).grid(row=0, column=5, sticky="w", padx=(6, 0))

        self.fig_spec = Figure(figsize=(6, 3), dpi=100)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")

        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.tab_spec)
        self.canvas_spec.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        toolbar_spec = NavigationToolbar2Tk(self.canvas_spec, self.tab_spec, pack_toolbar=False)
        toolbar_spec.update()
        toolbar_spec.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        # Cursor readout (shared)
        cursor_row = ttk.Frame(right)
        cursor_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        cursor_row.columnconfigure(0, weight=1)
        ttk.Label(cursor_row, textvariable=self.cursor_var).grid(row=0, column=0, sticky="w")

    def _set_defaults(self) -> None:
        here = Path(__file__).resolve().parent
        self.model_path_var.set(str(here / "kokoro-v1.0.onnx"))
        self.voices_path_var.set(str(here / "voices-v1.0.bin"))
        self.voice_var.set("af_sarah")
        self.lang_var.set("en-us")
        self.text.insert(
            "1.0",
            "Hello.\nThis audio was generated by Kokoro.\n\n"
            "Tip: Try different voices (af_*, am_*, bf_*, bm_*) and tweak speed."
        )
        self._update_speed_readout()
        self._plot_wave_empty("No waveform yet — generate audio to preview.")
        self._plot_spec_empty("No spectrogram yet — generate audio to preview.")
        self._set_levels_ui(None)

    def _update_speed_readout(self) -> None:
        self.speed_readout.configure(text=f"{self.speed_var.get():.2f}")

    # ---------------- File pickers ----------------

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Kokoro ONNX model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if p:
            self.model_path_var.set(p)

    def _browse_voices(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Kokoro voices bin",
            filetypes=[("Voices bin", "*.bin"), ("All files", "*.*")],
        )
        if p:
            self.voices_path_var.set(p)

    # ---------------- Kokoro loading ----------------

    def _ensure_kokoro(self) -> Kokoro:
        model_path = self.model_path_var.get().strip()
        voices_path = self.voices_path_var.get().strip()

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not voices_path or not os.path.isfile(voices_path):
            raise FileNotFoundError(f"Voices not found: {voices_path}")

        paths = (os.path.abspath(model_path), os.path.abspath(voices_path))
        if self._kokoro is None or self._kokoro_paths != paths:
            self.status_var.set("Loading model…")
            self.update_idletasks()
            self._kokoro = Kokoro(paths[0], paths[1])
            self._kokoro_paths = paths
        return self._kokoro

    # ---------------- Levels UI ----------------

    def _set_levels_ui(self, audio: Optional[GeneratedAudio]) -> None:
        if audio is None:
            self.rms_var.set("RMS: — dBFS")
            self.peak_var.set("Peak: — dBFS")
            self.meter_var.set(0)
            return

        self.rms_var.set(f"RMS: {audio.rms_dbfs:.1f} dBFS")
        self.peak_var.set(f"Peak: {audio.peak_dbfs:.1f} dBFS")

        # Meter mapping: -60 dBFS -> 0%, 0 dBFS -> 100%
        meter = int(_clamp((audio.rms_dbfs + 60.0) / 60.0, 0.0, 1.0) * 100.0)
        self.meter_var.set(meter)

    # ---------------- Plot helpers ----------------

    def _decimate_for_plot(self, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (decimated_y, stride) for plotting."""
        n = int(y.shape[0])
        if n <= self._plot_max_points:
            return y, 1
        stride = max(1, n // self._plot_max_points)
        return y[::stride], stride

    def _plot_wave_empty(self, msg: str) -> None:
        self.ax_wave.clear()
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.2)
        self.ax_wave.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax_wave.transAxes)
        self.canvas_wave.draw_idle()

    def _plot_spec_empty(self, msg: str) -> None:
        self.ax_spec.clear()
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax_spec.transAxes)
        self.canvas_spec.draw_idle()

    # ---------------- Waveform plotting ----------------

    def _update_waveform_plot(self) -> None:
        if not self._last_audio:
            self._plot_wave_empty("No waveform yet — generate audio to preview.")
            return

        y = self._last_audio.samples
        sr = self._last_audio.sample_rate
        total_secs = max(0.0, len(y) / float(sr))

        try:
            s0 = float(self._selection_start_var.get())
            s1 = float(self._selection_end_var.get())
        except Exception:
            messagebox.showerror("Invalid range", "Selection start/end must be numbers.")
            return

        if s1 <= s0:
            s0, s1 = 0.0, min(2.0, total_secs)
            self._selection_start_var.set(s0)
            self._selection_end_var.set(s1)

        s0 = _clamp(s0, 0.0, total_secs)
        s1 = _clamp(s1, 0.0, total_secs)
        if s1 <= s0:
            s0, s1 = 0.0, total_secs

        i0 = int(s0 * sr)
        i1 = int(s1 * sr)
        i0 = max(0, min(i0, len(y)))
        i1 = max(0, min(i1, len(y)))
        if i1 <= i0:
            i0, i1 = 0, len(y)

        seg = y[i0:i1]
        seg_plot, stride = self._decimate_for_plot(seg)

        t = (np.arange(seg_plot.shape[0], dtype=np.float32) * stride) / float(sr)
        t = t + (i0 / float(sr))

        self.ax_wave.clear()
        self.ax_wave.plot(t, seg_plot, linewidth=0.7)
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.2)
        self.ax_wave.set_title(self._last_audio.meta)

        self.canvas_wave.draw_idle()

    def _zoom_to_selection(self) -> None:
        self._update_waveform_plot()

    def _reset_view(self) -> None:
        if not self._last_audio:
            self._plot_wave_empty("No waveform yet — generate audio to preview.")
            return
        sr = self._last_audio.sample_rate
        total_secs = len(self._last_audio.samples) / float(sr)
        self._selection_start_var.set(0.0)
        self._selection_end_var.set(total_secs)
        self._update_waveform_plot()

    # ---------------- Spectrogram plotting ----------------

    def _auto_spec_params(self) -> None:
        """
        Pick reasonable NFFT/overlap for speech. Keeps it simple:
        - Prefer 1024 or 2048 depending on sample rate
        - Overlap ~75%
        """
        if not self._last_audio:
            self._spec_nfft_var.set(1024)
            self._spec_noverlap_var.set(768)
            return
        sr = self._last_audio.sample_rate
        nfft = 1024 if sr <= 24000 else 2048
        noverlap = int(nfft * 0.75)
        self._spec_nfft_var.set(int(nfft))
        self._spec_noverlap_var.set(int(noverlap))
        self._update_spectrogram_plot()

    def _update_spectrogram_plot(self) -> None:
        if not self._last_audio:
            self._plot_spec_empty("No spectrogram yet — generate audio to preview.")
            return

        y = self._last_audio.samples
        sr = self._last_audio.sample_rate

        try:
            nfft = int(self._spec_nfft_var.get())
            noverlap = int(self._spec_noverlap_var.get())
        except Exception:
            messagebox.showerror("Invalid params", "NFFT and Overlap must be integers.")
            return

        if nfft < 128:
            messagebox.showerror("Invalid params", "NFFT too small. Use >= 128.")
            return
        if noverlap < 0 or noverlap >= nfft:
            messagebox.showerror("Invalid params", "Overlap must be >= 0 and < NFFT.")
            return

        # Matplotlib specgram expects raw samples; returns (Pxx, freqs, bins, im)
        self.ax_spec.clear()
        self.ax_spec.set_title(self._last_audio.meta)
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")

        # For a cleaner plot, reduce dynamic range a bit via scale='dB'
        self.ax_spec.specgram(
            y,
            NFFT=nfft,
            Fs=sr,
            noverlap=noverlap,
            scale="dB",
        )

        self.canvas_spec.draw_idle()

    # ---------------- Cursor readout ----------------

    def _bind_matplotlib_cursor_events(self) -> None:
        # Waveform cursor
        self.canvas_wave.mpl_connect("motion_notify_event", self._on_wave_motion)
        # Spectrogram cursor
        self.canvas_spec.mpl_connect("motion_notify_event", self._on_spec_motion)

    def _on_wave_motion(self, event) -> None:
        if event.inaxes != self.ax_wave:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.cursor_var.set(f"Cursor (Wave): t={event.xdata:.3f}s, amp={event.ydata:.3f}")

    def _on_spec_motion(self, event) -> None:
        if event.inaxes != self.ax_spec:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.cursor_var.set(f"Cursor (Spec): t={event.xdata:.3f}s, f={event.ydata:.1f} Hz")

    # ---------------- Actions ----------------

    def _copy_last_meta(self) -> None:
        if not self._last_audio:
            messagebox.showinfo("Nothing to copy", "No audio generated yet.")
            return
        self.clipboard_clear()
        self.clipboard_append(self._last_audio.meta)
        self.status_var.set("Copied last meta to clipboard.")

    def _generate_clicked(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Busy", "Generation is already running.")
            return

        self.btn_generate.configure(state="disabled")
        self.status_var.set("Starting generation…")

        def _work() -> None:
            t0 = time.time()
            try:
                kokoro = self._ensure_kokoro()

                txt = self.text.get("1.0", "end").strip()
                if not txt:
                    raise ValueError("Text is empty.")

                voice = self.voice_var.get().strip()
                if not voice:
                    raise ValueError("Voice is empty.")

                lang = self.lang_var.get().strip() or "en-us"
                speed = float(self.speed_var.get())

                self._set_status_threadsafe("Generating audio…")
                samples, sr = kokoro.create(txt, voice=voice, speed=speed, lang=lang)
                samples = np.asarray(samples, dtype=np.float32).reshape(-1)

                rms_dbfs, peak_dbfs = _compute_levels_dbfs(samples)
                secs = len(samples) / float(sr)
                meta = f"voice={voice}, lang={lang}, speed={speed:.2f}, sr={int(sr)}, secs={secs:.2f}"

                audio = GeneratedAudio(
                    samples=samples,
                    sample_rate=int(sr),
                    meta=meta,
                    rms_dbfs=float(rms_dbfs),
                    peak_dbfs=float(peak_dbfs),
                )
                self._last_audio = audio

                # Set default zoom window
                self._set_selection_defaults_threadsafe(secs)

                dt = time.time() - t0
                self._set_last_meta_threadsafe(meta)
                self._set_status_threadsafe(f"Done in {dt:.2f}s. {meta}")

                # Update UI (plots + levels)
                self.after(0, lambda: self._set_levels_ui(audio))
                self.after(0, self._update_waveform_plot)
                self.after(0, self._update_spectrogram_plot)

                mode = self.output_mode.get()
                if mode == "play":
                    if not _HAS_SD:
                        self._set_status_threadsafe("Generated OK. (Playback disabled: install sounddevice.)")
                    else:
                        self._play_audio(samples, int(sr))
                elif mode == "wav":
                    out_path = str(Path(__file__).resolve().parent / "kokoro_out.wav")
                    _write_wav_pcm16(out_path, samples, int(sr))
                    self._set_status_threadsafe(f"Saved: {out_path}")
                else:
                    self._set_status_threadsafe("Generated OK.")
            except Exception as e:
                self._set_status_threadsafe(f"Error: {e}")
                self._show_error_threadsafe("Generation failed", str(e))
            finally:
                self._enable_generate_threadsafe()

        self._worker = threading.Thread(target=_work, daemon=True)
        self._worker.start()

    def _set_selection_defaults_threadsafe(self, total_secs: float) -> None:
        def _u() -> None:
            if total_secs <= 2.0:
                self._selection_start_var.set(0.0)
                self._selection_end_var.set(max(0.0, total_secs))
            else:
                self._selection_start_var.set(0.0)
                self._selection_end_var.set(2.0)

        self.after(0, _u)

    def _enable_generate_threadsafe(self) -> None:
        self.after(0, lambda: self.btn_generate.configure(state="normal"))

    def _set_status_threadsafe(self, s: str) -> None:
        self.after(0, lambda: self.status_var.set(s))

    def _set_last_meta_threadsafe(self, s: str) -> None:
        self.after(0, lambda: self.last_meta_var.set(s))

    def _show_error_threadsafe(self, title: str, msg: str) -> None:
        self.after(0, lambda: messagebox.showerror(title, msg))

    def _play_audio(self, samples: np.ndarray, sr: int) -> None:
        if not _HAS_SD:
            raise RuntimeError("sounddevice is not installed.")
        self._set_status_threadsafe("Playing audio…")
        sd.stop()
        sd.play(samples, sr)
        sd.wait()
        self._set_status_threadsafe("Playback finished.")

    def _stop_playback(self) -> None:
        if _HAS_SD:
            sd.stop()
            self.status_var.set("Playback stopped.")
        else:
            self.status_var.set("Playback disabled (sounddevice not installed).")

    def _play_last(self) -> None:
        if not self._last_audio:
            messagebox.showinfo("No audio", "No audio generated yet.")
            return
        if not _HAS_SD:
            messagebox.showinfo("Playback disabled", "Install sounddevice to enable playback:\n  pip install sounddevice")
            return
        try:
            self._play_audio(self._last_audio.samples, self._last_audio.sample_rate)
        except Exception as e:
            messagebox.showerror("Playback failed", str(e))

    def _save_last(self) -> None:
        if not self._last_audio:
            messagebox.showinfo("No audio", "No audio generated yet.")
            return

        p = filedialog.asksaveasfilename(
            title="Save WAV",
            defaultextension=".wav",
            initialfile="kokoro_out.wav",
            filetypes=[("WAV", "*.wav")],
        )
        if not p:
            return
        try:
            _write_wav_pcm16(p, self._last_audio.samples, self._last_audio.sample_rate)
            self.status_var.set(f"Saved: {p}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


if __name__ == "__main__":
    app = KokoroTestRig()
    app.mainloop()