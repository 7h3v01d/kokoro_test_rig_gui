# Kokoro-82M ONNX Test Rig (GUI)

A small, offline-friendly **GUI test harness** for **Kokoro-82M-v1.0-ONNX** using the `kokoro-onnx` Python package.

It’s designed to quickly validate:
- Model loading (ONNX + voices bin)
- Voice selection + speed control
- Output to **speakers** or **WAV**
- Visual inspection via **waveform + spectrogram**
- Basic level sanity checks (**RMS / Peak dBFS**)

---

## Features

### Core
- ✅ Load local `kokoro-v1.0.onnx` + `voices-v1.0.bin`
- ✅ Enter text and synthesize speech (offline once files are local)
- ✅ Choose voice, language, speed
- ✅ Output mode:
  - **Play to speakers** (requires `sounddevice`)
  - **Save WAV** (PCM16, no extra audio libs required)

### Model File Handling (Built-In Downloader)
- ⬇️ **Download model files inside the GUI**
  - Downloads `kokoro-v1.0.onnx` + `voices-v1.0.bin`
  - Uses streaming download + progress bars
  - Writes to `*.part` then renames (reduces risk of partial/corrupt files)
- ✅ **Startup check**
  - If model/voices files are missing at the configured paths, the app prompts to download them automatically.

### Analysis / Preview
- 📈 **Waveform viewer**
  - Zoom by time range (Start/End seconds)
  - Reset to full clip
- 🌈 **Spectrogram viewer**
  - Configurable NFFT / overlap
  - “Auto params” for quick speech-friendly settings
- 🎯 **Cursor readout**
  - Waveform: time + amplitude
  - Spectrogram: time + frequency
- 🔊 **RMS + Peak dBFS**
  - Includes a simple RMS meter for quick level sanity checks

---

## Requirements

- Python 3.10+ recommended (3.11 works fine)
- Windows/Linux/macOS

Python packages:
- `kokoro-onnx`
- `numpy`
- `matplotlib`
- `sounddevice` *(optional, only for speaker playback)*

---

## Install

### Create a venv (recommended)
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
### Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```
### Install deps

```bash
pip install -U -r requirements.txt
```
or
```bash
pip install -U kokoro-onnx numpy matplotlib
pip install -U sounddevice
```
If you don’t install sounddevice, the app still works (WAV output + plots). Playback buttons will be disabled or will warn.

### Run
```bash
python kokoro_test_rig_gui.py
```
On first launch, if the model files are missing, the rig will offer to download them.


## Model Files

Place these two files next to `kokoro_test_rig_gui.py` (or browse to them in the GUI):

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

### Recommended workflow

1. Launch the GUI
2. Accept the startup download prompt (or click Download model files…)
3. Generate speech and validate output/plots


## Usage Notes
### Output Modes

- Play to speakers
    - Uses `sounddevice` to play the generated waveform.
- Save WAV
    - Writes mono PCM16 WAV using Python’s standard `wave` module.
    - Default quick-save target (if you choose output mode “WAV”) is:
          - `./kokoro_out.wav` (next to the script)

### Waveform Tab

- Set Start/End seconds → click Zoom
- Click Reset to view the whole clip
- Hover to see cursor time/amplitude

### Spectrogram Tab

- Click Auto params then Render (or just Render if already set)
- Hover to see cursor time/frequency

### Levels

- RMS dBFS gives a practical loudness sanity check
- Peak dBFS shows headroom / clipping risk
- Meter maps RMS:

    - ~-60 dBFS → 0%
    - 0 dBFS → 100%

## Troubleshooting
### “Model not found” / “Voices not found”

- Use Download model files… in the GUI
- Or use Browse… buttons to select the files manually

### Playback not working

- Install sounddevice:
```bash
pip install sounddevice
```
- If playback still fails, use Save WAV and play the output using a standard media player.

### Matplotlib/Tk errors

- Ensure matplotlib is installed:
```bash
pip install -U matplotlib
```
- Some minimal Python installs may not include Tk; use a standard Python distribution that includes Tk support.
  
### Project Layout (minimal)
```code
.
├── kokoro_test_rig_gui.py
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```
## Credits / Upstream

- Model + tooling ecosystem is provided by the Kokoro ONNX community and related upstream projects.
- This GUI rig is a lightweight tester built around the `kokoro-onnx` Python interface.

## License

This test rig script is intended to be used alongside the upstream model/package licenses.
If you publish this in a repo, consider adding your chosen license file (MIT/Apache-2.0/etc.) and verify compatibility with upstream dependencies.# kokoro_test_rig_gui
