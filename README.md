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

### Analysis / Preview
- 📈 **Waveform viewer** (zoom by time range)
- 🌈 **Spectrogram viewer** (tunable NFFT/overlap, includes “Auto params”)
- 🎯 **Cursor readout**
  - Waveform: time + amplitude
  - Spectrogram: time + frequency
- 🔊 **RMS / Peak dBFS readout** + a simple RMS meter

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
pip install -U kokoro-onnx numpy matplotlib
pip install -U sounddevice
```
If you don’t install sounddevice, the app still works (WAV output + plots). Playback buttons will be disabled or will warn.

## Model Files

Place these two files next to `kokoro_test_rig_gui.py` (or browse to them in the GUI):

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

#### Download (Windows PowerShell)

PowerShell’s curl is an alias to Invoke-WebRequest, so use:
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" `
  -OutFile "kokoro-v1.0.onnx"

```
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" `
  -OutFile "voices-v1.0.bin"
```
### Download (real curl on Windows)

If you have the actual curl binary:
```powershell
curl.exe -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl.exe -L -o voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
### Run
```bash
python kokoro_test_rig_gui.py
```
On first run, verify the GUI shows your model/voices paths correctly (defaults assume they are beside the script).

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

Troubleshooting
“Model not found” / “Voices not found”

Make sure:

- The files exist
- Paths are correct
- You selected the correct `.onnx` and `.bin`

### No audio playback

- Install `sounddevice`:
```bash
pip install sounddevice
```
- On Windows, you may also need a working audio backend. If playback still fails, use Save WAV and play the file in a standard media player.

### Matplotlib/Tk errors

- Ensure you installed matplotlib and your Python includes Tk support.
- Reinstall:
```bash
pip install -U matplotlib
```
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
