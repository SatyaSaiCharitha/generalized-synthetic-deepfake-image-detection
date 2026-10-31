import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

def main():
    # If running with a python that is not our venv and venv python exists, re-launch with venv
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        print(f"[*] Switching to isolated virtual environment: {VENV_PYTHON}")
        cmd = [str(VENV_PYTHON), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
        try:
            subprocess.run(cmd, cwd=str(BASE_DIR))
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")
        return

    # Otherwise run directly with uvicorn
    import uvicorn
    print("\n" + "=" * 60)
    print("   GENIMAGE DETECTOR — PROFESSIONAL CLIENT INTERFACE")
    print("   Benchmarked on Tiny-GenImage (ResNet-50 vs ConvNeXt vs EfficientNet-B3)")
    print("   Deploying Winner: ConvNeXt-Tiny (98.86% Test Accuracy)")
    print("   Local Web UI: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
