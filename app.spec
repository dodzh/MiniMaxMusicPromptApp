# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification for MiniMax Music Prompt Studio
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# Collect all resources from essential packages
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all('customtkinter')
dnd_datas, dnd_binaries, dnd_hiddenimports = collect_all('tkinterdnd2')
librosa_datas, librosa_binaries, librosa_hiddenimports = collect_all('librosa')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')

all_datas = ctk_datas + dnd_datas + librosa_datas + soundfile_datas
all_binaries = ctk_binaries + dnd_binaries + librosa_binaries + soundfile_binaries
all_hiddenimports = (
    ctk_hiddenimports
    + dnd_hiddenimports
    + librosa_hiddenimports
    + soundfile_hiddenimports
    + [
        'scipy.special.cython_special',
        'scipy.spatial.transform._rotation_groups',
        'numba',
        'llvmlite',
        'lazy_loader',
        'audioread',
        'soxr',
        'requests',
        'json',
        'threading',
    ]
)

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'torch', 'transformers', 'notebook', 'IPython', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MiniMaxMusicPromptApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed dark-mode application (no black console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

