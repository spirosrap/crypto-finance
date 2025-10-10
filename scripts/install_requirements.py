#!/usr/bin/env python3
"""
Automates dependency installation for the crypto-finance project.

Steps performed:
1. Detects whether the TA-Lib C library is present on the system.
2. If missing, downloads and builds TA-Lib v0.6.4 into the chosen prefix (default: ~/.local).
3. Invokes `pip install -r requirements.txt` with the include/library paths exported so the
   Python `TA-Lib` wheel builds against the freshly installed library.

This script keeps the workflow consistent across Linux and macOS while avoiding manual setup.
For Windows or platforms without build tooling, the script exits with guidance instead of failing silently.
"""

from __future__ import annotations

import argparse
import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping


TA_LIB_VERSION = "0.6.4"
TA_LIB_SRC_FILENAME = f"ta-lib-{TA_LIB_VERSION}-src.tar.gz"
TA_LIB_BASE_URL = (
    f"https://github.com/TA-Lib/ta-lib/releases/download/v{TA_LIB_VERSION}/{TA_LIB_SRC_FILENAME}"
)


class InstallationError(RuntimeError):
    """Raised when automatic installation cannot be completed."""


def run_command(
    cmd: Iterable[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Execute a command and stream output, raising on failure."""
    printable_cwd = f" (cwd={cwd})" if cwd else ""
    print(f"[install] Running: {' '.join(cmd)}{printable_cwd}")
    subprocess.check_call(list(cmd), cwd=cwd, env=env)


def has_talib(prefix: Path) -> bool:
    """Return True if TA-Lib headers and libraries appear to be available."""
    lib_dir = prefix / "lib"
    include_dir = prefix / "include" / "ta-lib"
    lib_candidates = [
        "libta-lib.so",
        "libta-lib.so.0",
        "libta_lib.dylib",
        "ta_lib.dll",
    ]

    if include_dir.exists() and include_dir.is_dir():
        for lib_name in lib_candidates:
            if (lib_dir / lib_name).exists():
                return True

    # Fallback to dynamic loader lookup for users with system-wide installs.
    found_name = ctypes.util.find_library("ta_lib")
    if found_name:
        print(f"[install] Detected TA-Lib via system loader: {found_name}")
        return True

    return False


def download_talib_tarball(destination: Path) -> None:
    """Download the TA-Lib source tarball to the destination path."""
    if destination.exists():
        print(f"[install] Reusing existing archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[install] Downloading TA-Lib source from {TA_LIB_BASE_URL}")
    with urllib.request.urlopen(TA_LIB_BASE_URL) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def extract_tarball(archive_path: Path, target_dir: Path) -> Path:
    """Extract the tarball and return the path to the extracted source directory."""
    with tarfile.open(archive_path) as tar:
        root_members = {member.name.split("/")[0] for member in tar.getmembers() if "/" in member.name}
        tar.extractall(path=target_dir)

    if len(root_members) != 1:
        raise InstallationError("Unexpected TA-Lib archive layout.")

    extracted_dir = target_dir / root_members.pop()
    print(f"[install] Extracted TA-Lib sources to {extracted_dir}")
    return extracted_dir


def install_talib_linux(prefix: Path) -> None:
    """Compile and install TA-Lib on Linux."""
    build_root = Path("build") / "ta-lib"
    build_root.mkdir(parents=True, exist_ok=True)
    archive_path = build_root / TA_LIB_SRC_FILENAME
    download_talib_tarball(archive_path)

    with tempfile.TemporaryDirectory(dir=build_root) as tmp_dir:
        tmp_path = Path(tmp_dir)
        extracted = extract_tarball(archive_path, tmp_path)
        env = os.environ.copy()
        env.setdefault("CFLAGS", "")
        env.setdefault("LDFLAGS", "")

        run_command(["./configure", f"--prefix={prefix}"], cwd=extracted, env=env)
        run_command(["make"], cwd=extracted, env=env)
        run_command(["make", "install"], cwd=extracted, env=env)

    print(f"[install] TA-Lib installed under {prefix}")


def install_talib_macos(prefix: Path) -> None:
    """Install TA-Lib on macOS, relying on Homebrew when available."""
    brew = shutil.which("brew")
    if not brew:
        raise InstallationError(
            "Homebrew is required to install TA-Lib automatically on macOS. "
            "Install Homebrew from https://brew.sh and rerun this script."
        )

    run_command([brew, "install", "ta-lib"])

    # Homebrew installs into /usr/local or /opt/homebrew; ensure the prefix mirrors that path.
    # Symlink headers/libs into the chosen prefix to keep downstream logic consistent.
    brew_prefix = Path(subprocess.check_output([brew, "--prefix"]).decode().strip())
    header_source = brew_prefix / "include" / "ta-lib"
    lib_source = brew_prefix / "lib"

    prefix_include = prefix / "include" / "ta-lib"
    prefix_lib = prefix / "lib"
    prefix_include.parent.mkdir(parents=True, exist_ok=True)
    prefix_lib.mkdir(parents=True, exist_ok=True)

    if header_source.exists():
        if prefix_include.exists():
            shutil.rmtree(prefix_include)
        shutil.copytree(header_source, prefix_include)

    for candidate in lib_source.glob("libta_lib*.dylib"):
        shutil.copy2(candidate, prefix_lib / candidate.name)

    print(f"[install] Copied Homebrew TA-Lib artifacts into {prefix}")


def ensure_talib(prefix: Path) -> None:
    """Ensure TA-Lib is installed for the current platform."""
    if has_talib(prefix):
        print("[install] TA-Lib already present, skipping compilation.")
        return

    system = platform.system()
    print(f"[install] TA-Lib not detected. Installing for platform: {system}")

    if system == "Linux":
        install_talib_linux(prefix)
    elif system == "Darwin":
        install_talib_macos(prefix)
    else:
        raise InstallationError(
            f"Automatic TA-Lib installation is not implemented for {system}. "
            "Install the TA-Lib C library manually and rerun this script."
        )


def install_python_requirements(requirements_file: Path, prefix: Path) -> None:
    """Invoke pip to install the project requirements with TA-Lib include/lib hints."""
    env = os.environ.copy()
    include_dir = prefix / "include"
    lib_dir = prefix / "lib"

    env.setdefault("TA_INCLUDE_PATH", str(include_dir))
    env.setdefault("TA_LIBRARY_PATH", str(lib_dir))

    system = platform.system()
    if system == "Linux":
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    elif system == "Darwin":
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)

    run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install project requirements with automatic TA-Lib setup.")
    parser.add_argument(
        "-r",
        "--requirements",
        default="requirements.txt",
        type=Path,
        help="Path to the pip requirements file (default: requirements.txt).",
    )
    parser.add_argument(
        "--prefix",
        default=Path.home() / ".local",
        type=Path,
        help="Installation prefix for TA-Lib headers and libraries (default: ~/.local).",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="Only ensure TA-Lib is installed; do not run pip install.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix.expanduser().resolve()
    requirements_file = args.requirements.resolve()

    prefix.mkdir(parents=True, exist_ok=True)
    print(f"[install] Using TA-Lib installation prefix: {prefix}")

    try:
        ensure_talib(prefix)
    except InstallationError as exc:
        print(f"[install] ERROR: {exc}")
        sys.exit(1)

    if args.skip_pip:
        print("[install] Skipping pip install as requested.")
        return

    if not requirements_file.exists():
        print(f"[install] Requirements file not found: {requirements_file}")
        sys.exit(1)

    install_python_requirements(requirements_file, prefix)
    print("[install] Dependency installation complete.")


if __name__ == "__main__":
    main()
