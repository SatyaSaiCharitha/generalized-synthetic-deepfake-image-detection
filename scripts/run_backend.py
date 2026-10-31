import os
import sys
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

def main():
    if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        print(f"[*] Re-launching through project virtual environment: {VENV_PYTHON}")
        cmd = [str(VENV_PYTHON), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
        try:
            subprocess.run(cmd, cwd=str(BASE_DIR))
        except KeyboardInterrupt:
            print("\n[*] Server shutdown.")
        return

    import uvicorn
    os.chdir(str(BASE_DIR))
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
