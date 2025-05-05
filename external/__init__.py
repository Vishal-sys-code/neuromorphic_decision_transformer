"""External modules package."""

import sys
from pathlib import Path

# Add decision-transformer to Python path
dt_path = str(Path(__file__).parent / "decision-transformer")
if dt_path not in sys.path:
    sys.path.append(dt_path) 