#!/usr/bin/env python3
import os
import signal
import subprocess
import time

out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, errors="replace")
targets = []
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_s, _, args = line.partition(" ")
    try:
        pid = int(pid_s)
    except ValueError:
        continue
    if pid == os.getpid():
        continue
    if "run_gpt55_endpoint_gate.sh" in args or ("track2_multi_agent_plus.py" in args and "gpt-5.5" in args):
        targets.append((pid, args[:220]))
for pid, _ in targets:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
time.sleep(2)
print("stopped", len(targets))
for pid, args in targets:
    print(pid, args)
