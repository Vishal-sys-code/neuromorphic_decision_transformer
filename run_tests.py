import sys
import os
import pytest

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'snn-dt')))

if __name__ == "__main__":
    # Discover and run tests
    pytest.main()