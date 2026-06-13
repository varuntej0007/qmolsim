"""
run_all.py
Start both QMolSim APIs in one command.
Usage: python run_all.py
"""
import subprocess
import sys
import os
import time
import signal

processes = []

def shutdown(sig, frame):
    print("\nShutting down QMolSim...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

print("""
╔══════════════════════════════════════════════════════╗
║          QMolSim — Pharmaceutical Intelligence       ║
║          Quantum-Enhanced Drug & API Analysis        ║
╠══════════════════════════════════════════════════════╣
║  Main API  → http://0.0.0.0:5000                    ║
║  MSN API   → http://0.0.0.0:5001                    ║
║  Dashboard → http://0.0.0.0:5001/msn                ║
║  Executive → http://0.0.0.0:5001/executive          ║
╚══════════════════════════════════════════════════════╝
""")

os.makedirs("reports", exist_ok=True)
os.makedirs("phase2/data", exist_ok=True)
os.makedirs("phase3/data", exist_ok=True)

p1 = subprocess.Popen([sys.executable, "phase3/api.py"])
processes.append(p1)
time.sleep(2)

p2 = subprocess.Popen([sys.executable, "msn/msn_api.py"])
processes.append(p2)

print("Both APIs running. Press Ctrl+C to stop.\n")

for p in processes:
    p.wait()
