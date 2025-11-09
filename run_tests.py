import sys
from pathlib import Path
import pytest

# Add the snn-dt directory to the Python path
snn_dt_root = Path(__file__).resolve().parent / "snn-dt"
sys.path.insert(0, str(snn_dt_root))

# Run pytest
if __name__ == "__main__":
    sys.exit(pytest.main())