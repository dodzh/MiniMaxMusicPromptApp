# MiniMax Music Prompt Studio 🎵

A simmple application that enables you to create custom MiniMax Music prompts by simply dragging and dropping audio files. The application uses **Librosa** for audio analysis and **Ollama** for generating structured prompts. allows you to drag and drop MP3 and audio tracks, automatically extracts musical intelligence (BPM tempo, musical key, energy, spectral brightness, duration) with **Librosa**, and connects to your local **Ollama** LLM to generate structured, production-ready **MiniMax Music** prompts.

---

## ✨ Features

- **🎨 Modern Dark-Mode GUI**: High-contrast, sleek interface with customizable accent colors powered by CustomTkinter.
- **📥 Native Drag-and-Drop**: Drag & drop any audio file (`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`) directly onto the dropzone or use the file browser.
- **⚡ Librosa Audio Analysis**:
  - **Tempo (BPM)**: Precise beat tracking with harmonic/percussive separation.
  - **Musical Key**: Chroma analysis predicting the key and scale (e.g., *C Major*, *A Minor*).
  - **Energy & Dynamics**: RMS energy classification (*Chill*, *Balanced*, *Energetic*).
  - **Timbre & Brightness**: Spectral centroid analysis (*Warm/Mellow*, *Balanced*, *Bright/Crisp*).
  - **Track Duration**: Formatted time calculation.
- **🦙 Local Ollama Integration**:
  - Auto-detects running Ollama models (e.g., `llama3.2`, `mistral`, `qwen2.5`, `deepseek-r1`).
  - Streaming token responses in real-time.
  - Multi-preset templates: Full Song with Lyrics, Instrumental / Beat Production, Style Tags Only, EDM / Club Track, Cinematic / Ambient Score, Acoustic / Singer-Songwriter.
  - Custom theme / artist guidance input field.
- **📋 One-Click Output Actions**:
  - Instant **Copy to Clipboard** with visual toast notification.
  - **Save to .txt** file export.
  - Live cancel button and progress indicators.
- **📦 Standalone .exe Compilation**: Bundles all TkDND binaries, CustomTkinter assets, and Librosa/SciPy hooks into a single standalone `.exe`.

---

## 🚀 Quick Start (Running from Python)

### 1. Requirements
Ensure Python 3.10+ is installed along with the dependencies:
```bash
pip install customtkinter tkinterdnd2 librosa soundfile requests pyinstaller
```

### 2. Make sure Ollama is running
Ensure your local Ollama instance is active:
```bash
ollama serve
ollama pull llama3.2
```

### 3. Launch the Application
```bash
python app.py
```

---

## 🔨 Compiling into a Standalone `.exe`

Run the automated build script:
```bash
python build_exe.py
```
Or with PyInstaller directly:
```bash
pyinstaller --noconfirm --clean app.spec
```

The compiled standalone executable will be generated at:
```
dist/MiniMaxMusicPromptApp.exe
```

---

## 📁 Project Structure

```
MiniMaxMusicPromptApp/
├── app.py               # Main CustomTkinter + TkinterDnD2 GUI Application
├── audio_analyzer.py    # Librosa Audio Analysis Engine (BPM, Key, Energy)
├── ollama_client.py     # Local Ollama REST Client & Prompt Engineering
├── test_app.py          # Unit Test Suite
├── app.spec             # PyInstaller Packaging Specification
├── build_exe.py         # Automated Executable Compilation Script
└── README.md            # Project Documentation
```

