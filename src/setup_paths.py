"""
Author: Vishal Pandey
Email: pandeyvishal.mlprof@gmail.com
"""
import sys
import os

# Add external/decision_transformer/gym to sys.path for imports
external_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'decision_transformer', 'gym'))
if external_path not in sys.path:
    sys.path.insert(0, external_path)
