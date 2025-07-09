#!/usr/bin/env python3
"""
Quick Execution Script for Phase1 Training
Run this to start the Phase1 integration training and see the novelty in action!
"""

import sys
import os
import subprocess

def main():
    print("🧠 Starting Phase1 Training: Adaptive Spiking Windows + SDT")
    print("=" * 60)
    
    try:
        # Run the comprehensive training script
        result = subprocess.run([
            sys.executable, "phase1_comprehensive_training.py"
        ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("\n📊 Training Output:")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print("\n🔍 Error Output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 1 hour")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        
    print("\n🎯 Phase1 Training Session Complete!")

if __name__ == "__main__":
    main()
