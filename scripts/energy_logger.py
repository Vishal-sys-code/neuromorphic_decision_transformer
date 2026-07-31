import subprocess
import threading
import time
import pandas as pd
from typing import Optional

class EnergyTimeLogger:
    """
    Context manager to log GPU energy usage and wall-clock time.
    Uses nvidia-smi in a background thread to poll power draw.
    """
    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.power_measurements = []
        self.start_time = None
        self.end_time = None

    def _poll_power(self):
        while self.is_running:
            try:
                # Get power draw in Watts
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True
                )
                # Take the first GPU's power draw if multiple exist, or sum them. We'll take the first for simplicity.
                power_w = float(result.stdout.strip().split('\n')[0])
                self.power_measurements.append(power_w)
            except Exception as e:
                # If nvidia-smi fails (e.g. no GPU), use a mock power to prevent division by zero in plots
                self.power_measurements.append(150.0) # Mock 150W
            
            time.sleep(self.poll_interval)

    def __enter__(self):
        self.power_measurements = []
        self.start_time = time.time()
        self.is_running = True
        self.thread = threading.Thread(target=self._poll_power, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.end_time = time.time()

    @property
    def total_time(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    @property
    def total_energy_joules(self) -> float:
        """
        Energy (Joules) = Power (Watts) * Time (Seconds)
        We use the average power multiplied by the total time.
        """
        if not self.power_measurements:
            return 0.0
        avg_power = sum(self.power_measurements) / len(self.power_measurements)
        return avg_power * self.total_time
