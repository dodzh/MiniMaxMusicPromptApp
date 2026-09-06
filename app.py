"""
MiniMax Music Prompt Studio
Main Application with CustomTkinter GUI, TkinterDnD2 Drag & Drop, Librosa Audio Analysis, and Ollama LLM Integration.
"""

import os
import sys
import re
import threading
import time
from typing import Optional, List
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES

from audio_analyzer import analyze_audio, AudioAnalysisResult
from ollama_client import (
    DEFAULT_OLLAMA_HOST,
    PROMPT_PRESETS,
    check_ollama_connection,
    fetch_available_models,
    stream_generate_prompt,
)

# Supported audio extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


class MiniMaxPromptApp(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        # Initialize TkinterDnD wrapper
        self.TkdndVersion = TkinterDnD._require(self)

        # Window Configuration
        self.title("MiniMax Music Prompt Studio — AI Audio Analysis & Prompt Generator")
        self.geometry("1180x820")
        self.minsize(980, 680)

        # Appearance & Theme
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # State Variables
        self.current_audio_path: Optional[str] = None
        self.current_analysis: Optional[AudioAnalysisResult] = None
        self.generation_thread: Optional[threading.Thread] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.stop_generation_event = threading.Event()
        self.ollama_connected = False
        self.available_models: List[str] = []

        # UI Layout Construction
        self._build_ui()

        # Initial background tasks: Check Ollama connection & populate models
        self.after(200, self._check_ollama_status_async)

    def _build_ui(self):
        # Configure main grid
        self.grid_columnconfigure(0, weight=0)  # Left sidebar (fixed width)
        self.grid_columnconfigure(1, weight=1)  # Right output panel (expands)
        self.grid_rowconfigure(1, weight=1)     # Content row expands

        # -------------------------------------------------------------
        # 1. TOP HEADER BAR
        # -------------------------------------------------------------
        self.header_frame = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=("#1f232a", "#161920"))
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        self.header_frame.grid_propagate(False)

        # Title and Tagline
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="🎵 MiniMax Music Prompt Studio",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=("#e2e8f0", "#f8fafc"),
        )
        self.title_label.pack(side="left", padx=(18, 8), pady=12)

        self.badge_label = ctk.CTkLabel(
            self.header_frame,
            text="v1.0 • Librosa & Ollama Powered",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=("#2d3748", "#242c3d"),
            text_color=("#94a3b8", "#cbd5e1"),
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.badge_label.pack(side="left", padx=4, pady=14)

        # Ollama Connection Indicator on right
        self.ollama_status_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.ollama_status_frame.pack(side="right", padx=16, pady=10)

        self.ollama_status_dot = ctk.CTkLabel(
            self.ollama_status_frame,
            text="●",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#eab308",  # Yellow initially
        )
        self.ollama_status_dot.pack(side="left", padx=(0, 5))

        self.ollama_status_text = ctk.CTkLabel(
            self.ollama_status_frame,
            text="Ollama: Connecting...",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#94a3b8",
        )
        self.ollama_status_text.pack(side="left", padx=(0, 8))

        self.refresh_conn_btn = ctk.CTkButton(
            self.ollama_status_frame,
            text="⟳ Refresh",
            width=70,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color=("#2b3240", "#242a38"),
            hover_color=("#374151", "#2e3748"),
            command=self._check_ollama_status_async,
        )
        self.refresh_conn_btn.pack(side="left")

        # -------------------------------------------------------------
        # 2. LEFT SIDEBAR (Controls, DnD Dropzone, Audio Badges, Settings)
        # -------------------------------------------------------------
        self.left_scrollable = ctk.CTkScrollableFrame(
            self,
            width=430,
            corner_radius=0,
            fg_color=("#171a21", "#12141a"),
            scrollbar_button_color=("#2d3748", "#242b38"),
        )
        self.left_scrollable.grid(row=1, column=0, sticky="nsew", padx=(0, 1), pady=0)

        # --- Drop Zone Frame ---
        self.drop_zone = ctk.CTkFrame(
            self.left_scrollable,
            corner_radius=12,
            border_width=2,
            border_color=("#3b82f6", "#2563eb"),
            fg_color=("#1e2430", "#181d27"),
        )
        self.drop_zone.pack(fill="x", padx=14, pady=(14, 10))

        # Register Drop Target for DnD
        self.drop_zone.drop_target_register(DND_FILES)
        self.drop_zone.dnd_bind("<<Drop>>", self._on_file_drop)
        self.drop_zone.dnd_bind("<<DragEnter>>", self._on_drag_enter)
        self.drop_zone.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        self.drop_icon = ctk.CTkLabel(
            self.drop_zone,
            text="📥",
            font=ctk.CTkFont(size=36),
        )
        self.drop_icon.pack(pady=(16, 4))

        self.drop_primary_label = ctk.CTkLabel(
            self.drop_zone,
            text="Drag & Drop MP3 Audio Here",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="#f8fafc",
        )
        self.drop_primary_label.pack(pady=(0, 2))

        self.drop_secondary_label = ctk.CTkLabel(
            self.drop_zone,
            text="Supports MP3, WAV, FLAC, OGG, M4A",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#94a3b8",
        )
        self.drop_secondary_label.pack(pady=(0, 10))

        self.browse_btn = ctk.CTkButton(
            self.drop_zone,
            text="📁 Browse Audio File",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=32,
            fg_color=("#2563eb", "#1d4ed8"),
            hover_color=("#1d4ed8", "#1e40af"),
            command=self._on_browse_file,
        )
        self.browse_btn.pack(pady=(0, 16))

        # --- Audio Feature Dashboard Card ---
        self.audio_card = ctk.CTkFrame(
            self.left_scrollable,
            corner_radius=10,
            fg_color=("#1a202c", "#161b24"),
            border_width=1,
            border_color=("#2d3748", "#222a36"),
        )
        self.audio_card.pack(fill="x", padx=14, pady=8)

        self.card_title = ctk.CTkLabel(
            self.audio_card,
            text="📊 Audio Intelligence Readout",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#93c5fd",
            anchor="w",
        )
        self.card_title.pack(fill="x", padx=12, pady=(10, 6))

        # Grid of feature badges
        self.badge_grid = ctk.CTkFrame(self.audio_card, fg_color="transparent")
        self.badge_grid.pack(fill="x", padx=10, pady=(0, 10))
        self.badge_grid.grid_columnconfigure((0, 1), weight=1)

        # File & Duration Badge
        self.badge_file = self._create_badge(self.badge_grid, row=0, col=0, title="TRACK", value="No file loaded")
        self.badge_dur = self._create_badge(self.badge_grid, row=0, col=1, title="DURATION", value="--:--")

        # Tempo (BPM) & Key Badges
        self.badge_tempo = self._create_badge(self.badge_grid, row=1, col=0, title="TEMPO (BPM)", value="-- BPM", highlight=True)
        self.badge_key = self._create_badge(self.badge_grid, row=1, col=1, title="ESTIMATED KEY", value="--")

        # Energy & Timbre Badges
        self.badge_energy = self._create_badge(self.badge_grid, row=2, col=0, title="ENERGY DYNAMICS", value="--")
        self.badge_timbre = self._create_badge(self.badge_grid, row=2, col=1, title="TIMBRE / TONE", value="--")

        # --- Ollama & Prompt Configuration Section ---
        self.config_card = ctk.CTkFrame(
            self.left_scrollable,
            corner_radius=10,
            fg_color=("#1a202c", "#161b24"),
            border_width=1,
            border_color=("#2d3748", "#222a36"),
        )
        self.config_card.pack(fill="x", padx=14, pady=8)

        self.config_title = ctk.CTkLabel(
            self.config_card,
            text="⚙️ Prompt & Ollama Setup",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#93c5fd",
            anchor="w",
        )
        self.config_title.pack(fill="x", padx=12, pady=(10, 6))

        # Ollama Host URL
        self.host_label = ctk.CTkLabel(self.config_card, text="Ollama Server URL:", font=ctk.CTkFont(size=11), anchor="w")
        self.host_label.pack(fill="x", padx=12, pady=(4, 1))

        self.host_entry = ctk.CTkEntry(
            self.config_card,
            font=ctk.CTkFont(size=12),
            height=30,
        )
        self.host_entry.insert(0, DEFAULT_OLLAMA_HOST)
        self.host_entry.pack(fill="x", padx=12, pady=(0, 6))

        # Model Selector
        self.model_label = ctk.CTkLabel(self.config_card, text="Ollama Model:", font=ctk.CTkFont(size=11), anchor="w")
        self.model_label.pack(fill="x", padx=12, pady=(4, 1))

        self.model_dropdown = ctk.CTkOptionMenu(
            self.config_card,
            values=["llama3.2", "mistral", "qwen2.5", "deepseek-r1"],
            font=ctk.CTkFont(size=12),
            height=30,
        )
        self.model_dropdown.pack(fill="x", padx=12, pady=(0, 6))

        # MiniMax Prompt Preset
        self.preset_label = ctk.CTkLabel(self.config_card, text="MiniMax Prompt Preset:", font=ctk.CTkFont(size=11), anchor="w")
        self.preset_label.pack(fill="x", padx=12, pady=(4, 1))

        self.preset_dropdown = ctk.CTkOptionMenu(
            self.config_card,
            values=PROMPT_PRESETS,
            font=ctk.CTkFont(size=12),
            height=30,
        )
        self.preset_dropdown.pack(fill="x", padx=12, pady=(0, 6))

        # Additional Guidance / Theme Input
        self.notes_label = ctk.CTkLabel(
            self.config_card,
            text="Custom Theme / Artist Style / Mood (Optional):",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.notes_label.pack(fill="x", padx=12, pady=(4, 1))

        self.notes_textbox = ctk.CTkTextbox(
            self.config_card,
            height=65,
            font=ctk.CTkFont(size=12),
            corner_radius=6,
        )
        self.notes_textbox.pack(fill="x", padx=12, pady=(0, 12))

        # --- Primary Action Button (Generate) ---
        self.generate_btn = ctk.CTkButton(
            self.left_scrollable,
            text="✨ Generate MiniMax Prompt",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            height=44,
            corner_radius=8,
            fg_color=("#10b981", "#059669"),
            hover_color=("#059669", "#047857"),
            command=self._on_generate_click,
            state="disabled",
        )
        self.generate_btn.pack(fill="x", padx=14, pady=(8, 14))

        # Progress bar (Indeterminate during generation)
        self.progress_bar = ctk.CTkProgressBar(self.left_scrollable, mode="indeterminate", height=6)
        self.progress_bar.set(0)

        # -------------------------------------------------------------
        # 3. RIGHT PANEL (Output Prompt Studio, Clipboard, Actions)
        # -------------------------------------------------------------
        self.right_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("#12141a", "#0d0f14"))
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=0, pady=0)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Output Action Bar
        self.action_bar = ctk.CTkFrame(self.right_frame, height=48, fg_color="transparent")
        self.action_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))

        self.output_heading = ctk.CTkLabel(
            self.action_bar,
            text="Generated MiniMax Music Prompt",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="#f8fafc",
        )
        self.output_heading.pack(side="left", pady=4)

        # Copy Feedback Tooltip / Text
        self.copy_feedback_label = ctk.CTkLabel(
            self.action_bar,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#34d399",
        )
        self.copy_feedback_label.pack(side="left", padx=12, pady=4)

        # Action Buttons on right
        self.clear_btn = ctk.CTkButton(
            self.action_bar,
            text="🗑 Clear",
            width=70,
            height=30,
            font=ctk.CTkFont(size=12),
            fg_color=("#374151", "#28303d"),
            hover_color=("#4b5563", "#374151"),
            command=self._on_clear_output,
        )
        self.clear_btn.pack(side="right", padx=(6, 0))

        self.save_btn = ctk.CTkButton(
            self.action_bar,
            text="💾 Save .txt",
            width=85,
            height=30,
            font=ctk.CTkFont(size=12),
            fg_color=("#2563eb", "#1d4ed8"),
            hover_color=("#1d4ed8", "#1e40af"),
            command=self._on_save_file,
        )
        self.save_btn.pack(side="right", padx=(6, 0))

        self.copy_tags_btn = ctk.CTkButton(
            self.action_bar,
            text="🏷 Copy Production TAGS",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            height=30,
            fg_color=("#8b5cf6", "#7c3aed"),
            hover_color=("#7c3aed", "#6d28d9"),
            command=self._on_copy_production_tags,
        )
        self.copy_tags_btn.pack(side="left", padx=(12, 0))

        self.copy_lyrics_btn = ctk.CTkButton(
            self.action_bar,
            text="🎤 Copy Lyrics",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            height=30,
            fg_color=("#f59e0b", "#d97706"),
            hover_color=("#d97706", "#b45309"),
            command=self._on_copy_lyrics,
        )
        self.copy_lyrics_btn.pack(side="left", padx=(6, 0))

        self.copy_btn = ctk.CTkButton(
            self.action_bar,
            text="📋 Copy all",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=30,
            fg_color=("#059669", "#047857"),
            hover_color=("#047857", "#065f46"),
            command=self._on_copy_clipboard,
        )
        self.copy_btn.pack(side="right", padx=(6, 0))

        # Main Output Textbox
        self.output_textbox = ctk.CTkTextbox(
            self.right_frame,
            font=ctk.CTkFont(family="Consolas", size=13),
            corner_radius=8,
            fg_color=("#1a1e27", "#13161f"),
            border_width=1,
            border_color=("#2e3748", "#202633"),
            wrap="word",
        )
        self.output_textbox.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 10))

        # Initial Placeholder in Output
        self._set_output_text(
            "=================================================================\n"
            "  MiniMax Music Prompt Studio\n"
            "=================================================================\n\n"
            "1. Drag and drop an MP3 or audio file into the left dropzone.\n"
            "2. Librosa will automatically analyze tempo (BPM), key, and dynamics.\n"
            "3. Select your Ollama model and prompt preset.\n"
            "4. Click 'Generate MiniMax Prompt' to create production-ready tags & lyrics!\n"
        )

        # -------------------------------------------------------------
        # 4. BOTTOM STATUS FOOTER
        # -------------------------------------------------------------
        self.footer_frame = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color=("#171a21", "#101217"))
        self.footer_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.footer_frame.grid_propagate(False)

        self.status_bar_label = ctk.CTkLabel(
            self.footer_frame,
            text="Ready. Drop an MP3 file to begin.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#94a3b8",
            anchor="w",
        )
        self.status_bar_label.pack(side="left", padx=16, fill="x", expand=True)

    def _create_badge(self, parent, row: int, col: int, title: str, value: str, highlight: bool = False):
        """Helper to create stylized metadata badge cards."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=6,
            fg_color=("#242c3d", "#1e2433") if not highlight else ("#1e3a5f", "#172e4d"),
            border_width=1,
            border_color=("#374151", "#2b3548") if not highlight else ("#3b82f6", "#2563eb"),
        )
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        lbl_title = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color="#94a3b8" if not highlight else "#93c5fd",
            anchor="w",
        )
        lbl_title.pack(fill="x", padx=8, pady=(4, 0))

        lbl_val = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold" if highlight else "normal"),
            text_color="#f8fafc" if not highlight else "#60a5fa",
            anchor="w",
        )
        lbl_val.pack(fill="x", padx=8, pady=(0, 4))
        return {"frame": card, "title": lbl_title, "value": lbl_val}

    def _update_badge(self, badge_dict, text: str):
        badge_dict["value"].configure(text=text)

    # -------------------------------------------------------------
    # Drag and Drop & File Selection Handlers
    # -------------------------------------------------------------
    def _on_drag_enter(self, event):
        self.drop_zone.configure(border_color="#10b981", fg_color=("#1c2e26", "#14251e"))
        self.drop_primary_label.configure(text="Release to Load Audio File!")

    def _on_drag_leave(self, event):
        self.drop_zone.configure(border_color=("#3b82f6", "#2563eb"), fg_color=("#1e2430", "#181d27"))
        self.drop_primary_label.configure(text="Drag & Drop MP3 Audio Here")

    def _on_file_drop(self, event):
        self._on_drag_leave(event)
        raw_data = event.data
        if not raw_data:
            return

        # Parse file paths safely (handling Windows space braces)
        matches = re.findall(r'\{([^}]+)\}|(\S+)', raw_data)
        file_paths = [m[0] or m[1] for m in matches]

        if not file_paths:
            return

        selected_file = file_paths[0]
        ext = os.path.splitext(selected_file)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            messagebox.showwarning(
                "Unsupported File Format",
                f"The file '{os.path.basename(selected_file)}' is not a recognized audio file.\n"
                f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}",
            )
            return

        self._process_audio_file(selected_file)

    def _on_browse_file(self):
        filetypes = [
            ("Audio Files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma"),
            ("MP3 Audio", "*.mp3"),
            ("WAV Audio", "*.wav"),
            ("All Files", "*.*"),
        ]
        chosen = filedialog.askopenfilename(title="Select Audio File", filetypes=filetypes)
        if chosen:
            self._process_audio_file(chosen)

    def _process_audio_file(self, file_path: str):
        """Initiate threaded Librosa analysis on chosen audio file."""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File does not exist:\n{file_path}")
            return

        self.current_audio_path = file_path
        filename = os.path.basename(file_path)

        # Update UI state for analysis in progress
        self._set_status(f"Analyzing '{filename}' with Librosa (Tempo, Key, Energy)...")
        self._update_badge(self.badge_file, filename[:24] + ("..." if len(filename) > 24 else ""))
        self._update_badge(self.badge_dur, "Analyzing...")
        self._update_badge(self.badge_tempo, "Calculating...")
        self._update_badge(self.badge_key, "Analyzing...")
        self._update_badge(self.badge_energy, "Analyzing...")
        self._update_badge(self.badge_timbre, "Analyzing...")

        self.generate_btn.configure(state="disabled", text="⏳ Analyzing Audio...")
        self.progress_bar.pack(fill="x", padx=14, pady=(0, 10))
        self.progress_bar.start()

        # Run analysis in background thread to avoid freezing GUI
        def worker():
            try:
                start_t = time.time()
                analysis = analyze_audio(file_path)
                elapsed = time.time() - start_t
                self.after(0, lambda: self._on_analysis_success(analysis, elapsed))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self._on_analysis_error(err_msg))

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    def _on_analysis_success(self, analysis: AudioAnalysisResult, elapsed: float):
        self.current_analysis = analysis
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        # Update Badges with rich information
        self._update_badge(self.badge_file, analysis.file_name[:24] + ("..." if len(analysis.file_name) > 24 else ""))
        self._update_badge(self.badge_dur, analysis.duration_formatted)
        self._update_badge(self.badge_tempo, f"⚡ {analysis.tempo_rounded} BPM ({analysis.tempo_category.split('/')[0].strip()})")
        self._update_badge(self.badge_key, f"🎼 {analysis.estimated_key}")
        self._update_badge(self.badge_energy, f"🔥 {analysis.energy_level.split('/')[0].strip()}")
        self._update_badge(self.badge_timbre, f"✨ {analysis.timbre_brightness.split('/')[0].strip()}")

        self.generate_btn.configure(state="normal", text="✨ Generate MiniMax Prompt")
        self._set_status(f"Analysis complete in {elapsed:.2f}s! Detected {analysis.tempo_rounded} BPM ({analysis.estimated_key}). Ready to generate prompt.")

    def _on_analysis_error(self, err_msg: str):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.generate_btn.configure(state="disabled", text="✨ Generate MiniMax Prompt")
        self._set_status(f"Analysis failed: {err_msg}")
        messagebox.showerror("Audio Analysis Error", f"Failed to analyze audio file:\n{err_msg}")

    # -------------------------------------------------------------
    # Ollama Connection & Model Fetching
    # -------------------------------------------------------------
    def _check_ollama_status_async(self):
        """Check Ollama connectivity and fetch models in background."""
        host = self.host_entry.get().strip() or DEFAULT_OLLAMA_HOST
        self.ollama_status_dot.configure(text_color="#eab308")
        self.ollama_status_text.configure(text="Ollama: Checking...")

        def worker():
            connected = check_ollama_connection(host)
            models = fetch_available_models(host) if connected else []
            self.after(0, lambda: self._update_ollama_status(connected, models))

        threading.Thread(target=worker, daemon=True).start()

    def _update_ollama_status(self, connected: bool, models: List[str]):
        self.ollama_connected = connected
        if connected:
            self.ollama_status_dot.configure(text_color="#10b981")  # Green
            self.ollama_status_text.configure(text="Ollama: Online")
            if models:
                self.available_models = models
                self.model_dropdown.configure(values=models)
                if self.model_dropdown.get() not in models:
                    self.model_dropdown.set(models[0])
            self._set_status(f"Connected to Ollama. {len(models)} models available.")
        else:
            self.ollama_status_dot.configure(text_color="#ef4444")  # Red
            self.ollama_status_text.configure(text="Ollama: Offline")
            self._set_status("Ollama server not found at specified host. Make sure 'ollama serve' is active.")

    # -------------------------------------------------------------
    # Prompt Generation Workflow
    # -------------------------------------------------------------
    def _on_generate_click(self):
        # If already generating, act as Cancel button
        if self.generation_thread and self.generation_thread.is_alive():
            self.stop_generation_event.set()
            self._set_status("Cancelling generation...")
            return

        if not self.current_analysis:
            messagebox.showwarning("No Audio File", "Please drag and drop or browse an audio file first.")
            return

        host = self.host_entry.get().strip() or DEFAULT_OLLAMA_HOST
        model = self.model_dropdown.get()
        preset = self.preset_dropdown.get()
        user_notes = self.notes_textbox.get("1.0", "end").strip()

        # Clear output textbox and prepare streaming
        self._set_output_text("")
        self._set_status(f"Generating MiniMax prompt with Ollama [{model}]...")

        self.stop_generation_event.clear()
        self.generate_btn.configure(
            text="⏹ Cancel Generation",
            fg_color=("#ef4444", "#dc2626"),
            hover_color=("#dc2626", "#b91c1c"),
        )
        self.progress_bar.pack(fill="x", padx=14, pady=(0, 10))
        self.progress_bar.start()

        def stream_callback(chunk: str):
            self.after(0, lambda: self._append_output_text(chunk))

        def worker():
            start_t = time.time()
            stream_generate_prompt(
                host=host,
                model=model,
                analysis=self.current_analysis,
                preset=preset,
                user_notes=user_notes,
                chunk_callback=stream_callback,
                stop_event=self.stop_generation_event,
            )
            elapsed = time.time() - start_t
            self.after(0, lambda: self._on_generation_finished(elapsed))

        self.generation_thread = threading.Thread(target=worker, daemon=True)
        self.generation_thread.start()

    def _on_generation_finished(self, elapsed: float):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.generate_btn.configure(
            text="✨ Generate MiniMax Prompt",
            fg_color=("#10b981", "#059669"),
            hover_color=("#059669", "#047857"),
            state="normal",
        )
        if self.stop_generation_event.is_set():
            self._set_status("Generation cancelled by user.")
        else:
            self._set_status(f"MiniMax prompt generated successfully in {elapsed:.2f}s!")

    # -------------------------------------------------------------
    # Output Helpers & Clipboard Actions
    # -------------------------------------------------------------
    def _set_output_text(self, text: str):
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("1.0", text)

    def _append_output_text(self, text: str):
        self.output_textbox.insert("end", text)
        self.output_textbox.see("end")

    def _on_copy_clipboard(self):
        text = self.output_textbox.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

        # Temporary visual feedback animation
        self.copy_feedback_label.configure(text="✅ Copied to Clipboard!")
        self.after(2200, lambda: self.copy_feedback_label.configure(text=""))
        self._set_status("MiniMax prompt copied to clipboard!")

    def _copy_section(self, section_text: str, feedback: str, status: str):
        """Copy a specific output section to the clipboard with visual feedback."""
        if not section_text.strip():
            messagebox.showinfo(
                "Nothing to Copy",
                "This section was not found in the generated output.\nPlease generate a prompt first.",
            )
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(section_text.strip())
            self.update()
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Could not copy to clipboard:\n{e}")
            return

        self.copy_feedback_label.configure(text=f"✅ {feedback}")
        self.after(2200, lambda: self.copy_feedback_label.configure(text=""))
        self._set_status(status)

    def _split_output_sections(self):
        """Split the generated output into (production_tags, lyrics) sections."""
        full_text = self.output_textbox.get("1.0", "end").strip()
        if not full_text:
            return None, None

        # Locate the lyrics heading (tolerant of '### [SONG STRUCTURE & LYRICS]' style variants)
        match = re.search(r"(?im)^\s*#*\s*\[?\s*SONG\s+STRUCTURE", full_text)
        if not match:
            return None, None

        # Clean headers from the segments before returning
        tags_section = re.sub(r"^\s*#*\s*\[\s*STRUCTURED\s+CAPTION\s*\]", "", full_text[:match.start()]).strip()
        lyrics_section = re.sub(r"^#*\s*\[\s*SONG\s+STRUCTURE\s+&\s+LYRICS\s*\]", "", full_text[match.start():]).strip()
        return tags_section, lyrics_section

    def _on_copy_production_tags(self):
        tags_section, _ = self._split_output_sections()
        if tags_section is None:
            messagebox.showinfo(
                "Nothing to Copy",
                "No '[STRUCTURED CAPTION]' / Production TAGS section was found in the output.\n"
                "Please generate a full prompt first.",
            )
            return
        self._copy_section(tags_section, "Production TAGS Copied!", "Production TAGS copied to clipboard!")

    def _clean_lyrics_for_copy(self, lyrics_section: str) -> str:
        """Strip metadata headers (e.g. '### [SONG STRUCTURE & LYRICS]') and any
        other markdown header lines so that only bracketed section markers
        ([intro], [verse], [pre-chorus], [chorus], ...) and the raw lyrics
        themselves are returned."""
        text = lyrics_section or ""
        # Remove markdown header lines. Section tags such as [intro], [verse]
        # do not start with '#' (they begin with '[') and are therefore kept.
        text = re.sub(r"(?m)^\s*#{1,6}.*$", "", text)
        text = text.strip()
        # Collapse runs of blank lines into a single separator for tidiness
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text

    def _on_copy_lyrics(self):
        _, lyrics_section = self._split_output_sections()
        if lyrics_section is None:
            messagebox.showinfo(
                "Nothing to Copy",
                "No '[SONG STRUCTURE & LYRICS]' section was found in the output.\n"
                "Please generate a full prompt first.",
            )
            return
        cleaned = self._clean_lyrics_for_copy(lyrics_section)
        if not cleaned:
            messagebox.showinfo(
                "Nothing to Copy",
                "No '[SONG STRUCTURE & LYRICS]' section was found in the output.\n"
                "Please generate a full prompt first.",
            )
            return
        self._copy_section(cleaned, "Lyrics Copied!", "Lyrics copied to clipboard!")

    def _on_save_file(self):
        text = self.output_textbox.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty Output", "There is no generated prompt to save.")
            return

        default_name = "minimax_prompt.txt"
        if self.current_analysis:
            base = os.path.splitext(self.current_analysis.file_name)[0]
            default_name = f"{base}_minimax_prompt.txt"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text File", "*.txt"), ("All Files", "*.*")],
            title="Save MiniMax Prompt",
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(text)
                self._set_status(f"Prompt saved to {os.path.basename(filepath)}")
                messagebox.showinfo("Saved", f"Prompt successfully saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{str(e)}")

    def _on_clear_output(self):
        self._set_output_text("")
        self._set_status("Output cleared.")

    def _set_status(self, text: str):
        self.status_bar_label.configure(text=text)


def main():
    app = MiniMaxPromptApp()
    app.mainloop()


if __name__ == "__main__":
    main()

