"""
Unit Tests for MiniMax Music Prompt Studio
Tests audio analysis, prompt generation formatting, and GUI instantiation.
"""

import os
import unittest
import numpy as np
import soundfile as sf

from audio_analyzer import analyze_audio, format_duration, categorize_tempo, AudioAnalysisResult
from ollama_client import build_user_prompt, PROMPT_PRESETS, SYSTEM_PROMPT


class TestAudioAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_wav = "test_synth_audio.wav"
        sr = 22050
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz Sine with 120 BPM percussive pulses (every 0.5s)
        signal = np.sin(2 * np.pi * 440 * t) * 0.1
        for beat_t in np.arange(0, duration, 0.5):
            idx = int(beat_t * sr)
            signal[idx:min(idx + 400, len(signal))] += 0.8

        sf.write(cls.test_wav, signal, sr)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_wav):
            os.remove(cls.test_wav)

    def test_duration_formatter(self):
        self.assertEqual(format_duration(65.0), "01:05")
        self.assertEqual(format_duration(180.0), "03:00")
        self.assertEqual(format_duration(9.5), "00:09")

    def test_tempo_categorization(self):
        self.assertEqual(categorize_tempo(60), "Largo / Very Slow")
        self.assertEqual(categorize_tempo(120), "Moderato / Mid-Tempo")
        self.assertEqual(categorize_tempo(140), "Allegro / Upbeat & Energetic")
        self.assertEqual(categorize_tempo(175), "Presto / High Speed")

    def test_audio_analysis(self):
        result = analyze_audio(self.test_wav)
        self.assertIsInstance(result, AudioAnalysisResult)
        self.assertAlmostEqual(result.duration_sec, 4.0, delta=0.5)
        self.assertGreater(result.tempo_bpm, 80)
        self.assertLess(result.tempo_bpm, 160)
        self.assertTrue("Major" in result.estimated_key or "Minor" in result.estimated_key)
        self.assertIsNotNone(result.energy_level)
        self.assertIsNotNone(result.timbre_brightness)

        summary = result.to_summary_dict()
        self.assertIn("Tempo (BPM)", summary)
        self.assertIn("Estimated Key", summary)


class TestOllamaClient(unittest.TestCase):
    def test_user_prompt_construction(self):
        dummy_result = AudioAnalysisResult(
            file_path="C:/test/song.mp3",
            file_name="song.mp3",
            duration_sec=180.0,
            duration_formatted="03:00",
            tempo_bpm=128.0,
            tempo_rounded=128,
            tempo_category="Moderato / Mid-Tempo",
            estimated_key="F# Minor",
            energy_level="High / Energetic",
            energy_rms=0.22,
            timbre_brightness="Bright / Crisp / Airy",
            spectral_centroid=3200.0,
            sample_rate=22050,
        )

        prompt = build_user_prompt(
            dummy_result,
            preset=PROMPT_PRESETS[0],
            user_notes="Cyberpunk theme with heavy analog bass",
        )

        self.assertIn("128.0 BPM", prompt)
        self.assertIn("F# Minor", prompt)
        self.assertIn("Cyberpunk theme", prompt)
        self.assertIn("MiniMax Music", prompt)


class TestGUIInstantiation(unittest.TestCase):
    def test_app_creation(self):
        from app import MiniMaxPromptApp
        app = MiniMaxPromptApp()
        app.withdraw()  # keep off-screen

        # Verify widgets exist
        self.assertIsNotNone(app.drop_zone)
        self.assertIsNotNone(app.model_dropdown)
        self.assertIsNotNone(app.output_textbox)
        self.assertIsNotNone(app.generate_btn)
        self.assertIsNotNone(app.copy_btn)

        app.destroy()


if __name__ == "__main__":
    unittest.main()
