"""
Audio Analyzer Module
Uses Librosa to extract musical and structural features from audio files (MP3, WAV, FLAC, OGG, etc.)
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import librosa


@dataclass
class AudioAnalysisResult:
    file_path: str
    file_name: str
    duration_sec: float
    duration_formatted: str
    tempo_bpm: float
    tempo_rounded: int
    tempo_category: str
    estimated_key: str
    energy_level: str
    energy_rms: float
    timbre_brightness: str
    spectral_centroid: float
    sample_rate: int

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "File": self.file_name,
            "Duration": self.duration_formatted,
            "Tempo (BPM)": f"{self.tempo_bpm:.1f} BPM ({self.tempo_category})",
            "Estimated Key": self.estimated_key,
            "Energy Dynamics": f"{self.energy_level} (RMS: {self.energy_rms:.3f})",
            "Timbre & Tone": f"{self.timbre_brightness} ({self.spectral_centroid:.0f} Hz)",
        }


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def categorize_tempo(bpm: float) -> str:
    """Categorize BPM into descriptive tempo markings."""
    if bpm < 70:
        return "Largo / Very Slow"
    elif bpm < 85:
        return "Adagio / Slow"
    elif bpm < 108:
        return "Andante / Walking Pace"
    elif bpm < 128:
        return "Moderato / Mid-Tempo"
    elif bpm < 145:
        return "Allegro / Upbeat & Energetic"
    elif bpm < 170:
        return "Vivace / Fast"
    else:
        return "Presto / High Speed"

def estimate_key_and_mode(y_harmonic: np.ndarray, sr: int) -> str:
    """Estimate the musical key and major/minor scale from isolated harmonic audio chroma with tuning correction."""
    try:
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=tuning)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)

    chroma_mean = np.mean(chroma, axis=1)
    notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_score = -np.inf
    best_key = "C Major"

    for i in range(12):
        maj_corr = np.corrcoef(np.roll(major_profile, i), chroma_mean)[0, 1]
        if maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{notes[i]} Major"

        min_corr = np.corrcoef(np.roll(minor_profile, i), chroma_mean)[0, 1]
        if min_corr > best_score:
            best_score = min_corr
            best_key = f"{notes[i]} Minor"

    return best_key


def analyze_audio(file_path: str, max_duration: Optional[float] = 120.0) -> AudioAnalysisResult:
    """
    Analyze an audio file and extract tempo, key, energy, and timbre features.
    Loads up to `max_duration` seconds for rapid analysis.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load audio (mono, 22.05 kHz standard for music analysis)
    # Using 120s max for swift processing of large audio tracks
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=max_duration)
    
    # Total file duration
    try:
        total_duration = librosa.get_duration(path=file_path)
    except Exception:
        total_duration = librosa.get_duration(y=y, sr=sr)

    # 1. Harmonic-Percussive Separation for more accurate beat tracking
    y_harmonic, y_percussive = librosa.effects.hpss(y)

# 2. Tempo / BPM Estimation with adjusted hop_length for transient smoothing
    hop_length = 512  # Try increasing to 1024 if acoustic tracks still double-count
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, hop_length=hop_length)
    
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    if hasattr(tempo, '__iter__'):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo_val = float(tempo)
    
    if tempo_val <= 0:
        tempo_val = 120.0

    tempo_rounded = int(round(tempo_val))
    tempo_cat = categorize_tempo(tempo_val)

    # 3. Key and Mode estimation
    est_key = estimate_key_and_mode(y_harmonic, sr)

    # 4. Energy (RMS)
    rms_arr = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms_arr))
    if rms_mean < 0.05:
        energy_lvl = "Low / Chill"
    elif rms_mean < 0.15:
        energy_lvl = "Moderate / Balanced"
    else:
        energy_lvl = "High / Energetic"

    # 5. Spectral Centroid (Timbre brightness)
    spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    if spec_cent < 1500:
        timbre = "Warm / Dark / Mellow"
    elif spec_cent < 3000:
        timbre = "Balanced / Natural"
    else:
        timbre = "Bright / Crisp / Airy"

    file_name = os.path.basename(file_path)

    return AudioAnalysisResult(
        file_path=file_path,
        file_name=file_name,
        duration_sec=total_duration,
        duration_formatted=format_duration(total_duration),
        tempo_bpm=tempo_val,
        tempo_rounded=tempo_rounded,
        tempo_category=tempo_cat,
        estimated_key=est_key,
        energy_level=energy_lvl,
        energy_rms=rms_mean,
        timbre_brightness=timbre,
        spectral_centroid=spec_cent,
        sample_rate=sr,
    )

