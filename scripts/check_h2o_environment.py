"""Check the local Java, Python, NumPy, and H2O environment."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import numpy as np


def main() -> None:
    """Print and validate H2O prerequisites."""
    java = subprocess.run(
        ["java", "-version"],
        capture_output=True,
        text=True,
        check=False,
    )

    java_output = (
        java.stderr.strip()
        or java.stdout.strip()
    )

    if java.returncode != 0:
        raise RuntimeError(
            "Java is unavailable. Install Java 17."
        )

    numpy_major = int(
        np.__version__.split(".")[0]
    )

    if numpy_major >= 2:
        raise RuntimeError(
            "H2O requires NumPy below version 2."
        )

    print(
        "Python:",
        sys.version.split()[0],
    )
    print(
        "NumPy:",
        np.__version__,
    )
    print(
        "H2O:",
        importlib.metadata.version("h2o"),
    )
    print(
        "Java:",
        java_output.splitlines()[0],
    )
    print("H2O ENVIRONMENT STATUS: PASS")


if __name__ == "__main__":
    main()
