# -*- coding: utf-8 -*-
"""Small process helper that avoids shell self-matching pitfalls."""

import argparse
import os
import signal
import subprocess
import time


def ps_rows():
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) == 5:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "stat": parts[2], "etime": parts[3], "cmd": parts[4]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-pattern", action="append", default=[])
    parser.add_argument("--list-pattern", action="append", default=[])
    args = parser.parse_args()
    me = os.getpid()
    parent = os.getppid()
    rows = ps_rows()
    if args.kill_pattern:
        for row in rows:
            if row["pid"] in {me, parent}:
                continue
            if any(p in row["cmd"] for p in args.kill_pattern):
                print(f"killing pid={row['pid']} cmd={row['cmd']}")
                try:
                    os.kill(row["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(1)
        rows = ps_rows()
        for row in rows:
            if row["pid"] in {me, parent}:
                continue
            if any(p in row["cmd"] for p in args.kill_pattern):
                print(f"killing -9 pid={row['pid']} cmd={row['cmd']}")
                try:
                    os.kill(row["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
    patterns = args.list_pattern or args.kill_pattern
    rows = ps_rows()
    for row in rows:
        if not patterns or any(p in row["cmd"] for p in patterns):
            print(f"{row['pid']:>8} {row['ppid']:>8} {row['stat']:<8} {row['etime']:<10} {row['cmd']}")


if __name__ == "__main__":
    main()
