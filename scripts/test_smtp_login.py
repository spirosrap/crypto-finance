#!/usr/bin/env python3
"""
Minimal SMTP login tester for the finder alerts.

Reads SMTP settings from environment (or .env if available), attempts a STARTTLS
login, and prints a clear success/failure message. No email is sent.
"""

from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path


def load_dotenv_if_available() -> bool:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return False
    try:
        return load_dotenv()
    except Exception:
        return False


def main() -> int:
    load_dotenv_if_available()

    host = os.getenv("FINDER_ALERT_SMTP_HOST") or os.getenv("SMTP_HOST") or ""
    port_raw = os.getenv("FINDER_ALERT_SMTP_PORT") or os.getenv("SMTP_PORT") or "587"
    user = os.getenv("FINDER_ALERT_SMTP_USER") or os.getenv("SMTP_USER") or ""
    password = os.getenv("FINDER_ALERT_SMTP_PASS") or os.getenv("SMTP_PASS") or ""
    starttls_flag = os.getenv("FINDER_ALERT_SMTP_STARTTLS", "1").lower() not in {"0", "false", "no"}

    def _exit(msg: str, code: int) -> int:
        print(msg)
        return code

    if not host or not user or not password:
        return _exit("Missing SMTP credentials: ensure HOST/USER/PASS are set in the environment or .env", 1)

    try:
        port = int(port_raw)
    except ValueError:
        return _exit(f"Invalid SMTP port '{port_raw}'", 1)

    print(f"SMTP host: {host}")
    print(f"SMTP user: {user}")
    print(f"SMTP port: {port}")
    print(f"Password length: {len(password)}")
    print(f"STARTTLS: {'enabled' if starttls_flag else 'disabled'}")

    import smtplib

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if starttls_flag:
                server.starttls(context=ctx)
            server.login(user, password)
        print("SMTP login succeeded.")
        return 0
    except Exception as exc:
        print(f"SMTP login failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
