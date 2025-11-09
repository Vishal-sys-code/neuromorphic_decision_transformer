import sys
import os
import pytest

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    sys.path.insert(0, os.path.abspath("snn-dt"))
    print("sys.path:", sys.path)
    sys.exit(pytest.main(["-x", "snn-dt/tests/test_models.py"]))