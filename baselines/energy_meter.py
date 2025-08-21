"""
baselines/energy_meter.py
Measure NVIDIA Jetson power via tegrastats.
"""
import subprocess, time, csv, os

class TegraMeter:
    def __init__(self, log_file):
        self.log_file = log_file
        self.proc = None
    def __enter__(self):
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", "100"],
            stdout=open(self.log_file, "w"),
        )
        return self
    def __exit__(self, *args):
        self.proc.terminate()
        self.proc.wait()

def parse_tegra_log(log_file):
    """Return average power in mW."""
    watts = []
    with open(log_file) as f:
        for line in f:
            try:
                tok = line.split()[-2].split("/")[0]
                watts.append(float(tok))
            except Exception:
                continue
    return sum(watts) / len(watts) if watts else 0.0
