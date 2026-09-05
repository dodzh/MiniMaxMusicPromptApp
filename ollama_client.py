"""
Ollama Client Module
Interacts with local Ollama instance to generate structured MiniMax Music prompts based on audio analysis.
"""

import json
from typing import List, Dict, Any, Optional, Callable
import requests
from audio_analyzer import AudioAnalysisResult

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

PROMPT_PRESETS = [
    "Full Song with Lyrics & Performance Cues",
    "Instrumental / Production Track",
    "Style Tags & Musical Blueprint Only",
    "EDM / Electronic Club Track",
    "Cinematic / Ambient Score",
    "Acoustic / Singer-Songwriter",
]

SYSTEM_PROMPT = """You are an expert music producer and prompt engineer specialized in AI music generation for MiniMax Music 3.
Your mission is to generate perfectly structured, production-ready prompts divided strictly into two inputs: a Structured Caption and Lyrical Input based on the official MiniMax Music 3 guidelines.

CRITICAL FORMATTING RULES:
1. Do NOT use any Markdown formatting whatsoever (no bolding '**', no italics '*', no blockquotes '>'). Output plain text only.
2. Format strictly into two main sections:
   ### [STRUCTURED CAPTION]
   (Which must contain three sub-headings: Global Metadata, Vocal Details, and Arrangement)
   ### [SONG STRUCTURE & LYRICS]
3. In the STRUCTURED CAPTION section, detail the song under these exact three headings:
   - Global Metadata (Basic Attributes, Global Emotional Progression, Application Scenarios & Imagery, Sonics & Production Profile)
   - Vocal Details (Vocal Gender & Timbre, Vocal Style, Harmony/Backing Vocals, Vocal FX)
   - Arrangement (Instrument Lifecycle, Groove & Foundation Progression, Embellishments & Textures)
4. In the [SONG STRUCTURE & LYRICS] section, every section tag (e.g., [intro], [verse], [pre-chorus], [chorus], [bridge], [outro]) MUST sit entirely on its own line.
5. NEVER place bar counts, tempo metrics, or instrumental mixing instructions inside the lyrics section. The lyrics section contains ONLY section tags and raw, plain-text words to be sung. No descriptive stage directions or bracketed production notes.
6. Use descriptive musical terminology only in the caption. Never output raw filenames, file paths, or internal technical metrics like RMS values or frequency Hz counts.
"""

def check_ollama_connection(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 3.0) -> bool:
    """Check if the Ollama server is accessible."""
    try:
        url = host.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_available_models(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 4.0) -> List[str]:
    """Fetch the list of installed model names from Ollama."""
    try:
        url = host.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return models if models else ["llama3.2", "mistral", "qwen2.5"]
    except Exception:
        pass
    return ["llama3.2", "llama3", "mistral", "qwen2.5", "deepseek-r1", "phi3"]


def build_user_prompt(
    analysis: AudioAnalysisResult,
    preset: str,
    user_notes: str = "",
) -> str:
    """Build the prompt sent to the LLM with audio analysis context."""
    notes_section = f"\nUser Additional Notes / Theme:\n\"{user_notes.strip()}\"\n" if user_notes.strip() else ""

    return f"""AUDIO ANALYSIS DATA:
- Source Track: {analysis.file_name}
- Exact Tempo: {analysis.tempo_bpm:.1f} BPM ({analysis.tempo_category})
- Estimated Key: {analysis.estimated_key}
- Energy Dynamics: {analysis.energy_level} (RMS: {analysis.energy_rms:.4f})
- Timbre & Brightness: {analysis.timbre_brightness} ({analysis.spectral_centroid:.0f} Hz)
- Duration: {analysis.duration_formatted}

REQUESTED FORMAT / PRESET:
{preset}
{notes_section}
Generate a complete, production-grade MiniMax Music prompt tailored specifically to this BPM ({analysis.tempo_rounded} BPM), Key ({analysis.estimated_key}), and energy profile. Include rich style tags and structured sections.
"""


def stream_generate_prompt(
    host: str,
    model: str,
    analysis: AudioAnalysisResult,
    preset: str,
    user_notes: str,
    chunk_callback: Callable[[str], None],
    stop_event: Optional[Any] = None,
) -> str:
    """
    Stream prompt generation from Ollama via `/api/generate`.
    Calls `chunk_callback(chunk_text)` on each token received.
    """
    url = host.rstrip("/") + "/api/generate"
    prompt_text = build_user_prompt(analysis, preset, user_notes)

    payload = {
        "model": model,
        "prompt": prompt_text,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {
            "temperature": 0.75,
            "top_p": 0.9,
        },
    }

    full_response = []
    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        for line in response.iter_lines():
            if stop_event and stop_event.is_set():
                break
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                text_piece = chunk.get("response", "")
                if text_piece:
                    full_response.append(text_piece)
                    chunk_callback(text_piece)
                if chunk.get("done", False):
                    break
            except Exception:
                continue

    except requests.exceptions.ConnectionError:
        err = f"\n[Error: Unable to connect to Ollama at {host}. Please ensure Ollama is running ('ollama serve').]"
        chunk_callback(err)
        return err
    except Exception as e:
        err = f"\n[Error during generation: {str(e)}]"
        chunk_callback(err)
        return err

    return "".join(full_response)

