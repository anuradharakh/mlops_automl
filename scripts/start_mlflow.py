"""Start the local MLflow UI."""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
    ]
    print("Open http://127.0.0.1:5000")
    raise SystemExit(subprocess.call(command))
