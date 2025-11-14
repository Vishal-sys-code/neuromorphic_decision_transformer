import sys
import pytest
from pathlib import Path

# Add the snn-dt directory to the Python path
snn_dt_path = str(Path(__file__).resolve().parent / "snn-dt")
if snn_dt_path not in sys.path:
    sys.path.insert(0, snn_dt_path)

# Run pytest
sys.exit(pytest.main(["-s", "snn-dt/tests/"]))