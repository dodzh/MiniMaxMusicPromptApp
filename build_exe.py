"""
Build Script for MiniMax Music Prompt Studio Standalone Executable
Compiles the application into a standalone .exe in the dist/ folder.
"""

import os
import sys
import subprocess
import shutil

def build():
    print("=" * 60)
    print("Building MiniMax Music Prompt Studio Standalone Executable")
    print("=" * 60)

    # Base workspace directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    spec_file = os.path.join(base_dir, "app.spec")
    if not os.path.exists(spec_file):
        print(f"Error: Spec file not found: {spec_file}")
        sys.exit(1)

    # Clean old build artifacts
    for folder in ["build", "dist"]:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            print(f"Cleaning {folder}/...")
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"Warning: Could not remove {folder_path}: {e}")

    # Run PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "app.spec",
    ]

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[BUILD FAILED] PyInstaller returned code {result.returncode}")
        sys.exit(result.returncode)

    exe_path = os.path.join(base_dir, "dist", "MiniMaxMusicPromptApp.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL!")
        print(f"Executable created: {exe_path}")
        print(f"File Size: {size_mb:.2f} MB")
        print("=" * 60)
    else:
        print(f"\n[ERROR] Executable was not found at {exe_path}")
        sys.exit(1)

if __name__ == "__main__":
    build()

