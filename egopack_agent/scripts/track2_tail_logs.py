#!/usr/bin/env python3
import re
import sys
from pathlib import Path


TOKEN_RE = re.compile(r"sk-[A-Za-z0-9_-]+")


def red(text: str) -> str:
    return TOKEN_RE.sub("sk-REDACTED", text)


def safe_print(text: str) -> None:
    sys.stdout.buffer.write(text.encode("ascii", "backslashreplace") + b"\n")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: track2_tail_logs.py <log> [<log> ...]")
        return 2
    for arg in sys.argv[1:]:
        path = Path(arg)
        safe_print(f"==== {path} ====")
        if not path.exists():
            safe_print("missing")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        safe_print("-- head --")
        safe_print(red("\n".join(lines[:180])))
        safe_print("-- tail --")
        safe_print(red("\n".join(lines[-100:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
