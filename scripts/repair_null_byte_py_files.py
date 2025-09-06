#!/usr/bin/env python3
# scripts/repair_null_byte_py_files.py
# Scan DSF-related .py files (external and src) for null bytes and re-encode to UTF-8 safely.
# Usage: python scripts/repair_null_byte_py_files.py

import sys
import os
import shutil
from pathlib import Path

def find_target_paths(repo_root: Path):
    targets = []
    ext_models = repo_root / "external" / "DecisionSpikeFormer" / "gym" / "models"
    src_models = repo_root / "src" / "models" / "dsf_models"
    for d in (ext_models, src_models):
        if d.exists() and d.is_dir():
            for p in sorted(d.glob("*.py")):
                targets.append(p)
    return targets

def has_null_bytes(p: Path) -> bool:
    try:
        b = p.read_bytes()
        return b.find(b"\x00") != -1
    except Exception as e:
        print(f"  [ERROR] Could not read {p}: {e}")
        return False

def backup_file(p: Path) -> Path:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        try:
            shutil.copy2(p, bak)
            print(f"  + backup: {bak}")
        except Exception as e:
            print(f"  ! backup failed for {p}: {e}")
    else:
        print(f"  - backup exists: {bak}")
    return bak

def try_decode_bytes(b: bytes):
    enc_candidates = ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1", "cp1252"]
    for enc in enc_candidates:
        try:
            text = b.decode(enc)
            return text, enc
        except Exception:
            continue
    return None, None

def repair_file(p: Path) -> bool:
    print(f"Checking: {p}")
    try:
        raw = p.read_bytes()
    except Exception as e:
        print(f"  [ERROR] cannot read file: {e}")
        return False

    if b"\x00" not in raw:
        print("  OK: no null bytes found.")
        return True

    print("  Found null bytes. Attempting repair...")
    backup_file(p)

    text, enc = try_decode_bytes(raw)
    if text is None:
        print("  [FAIL] could not decode using common encodings. Showing hexdump head (200 bytes):")
        head = raw[:200]
        print("  " + head.hex())
        return False

    # Write to temp and atomically replace
    tmp = p.with_name(p.name + ".fixed")
    try:
        tmp.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"  [ERROR] writing fixed file failed: {e}")
        return False

    try:
        os.replace(str(tmp), str(p))
        print(f"  Replaced {p} (decoded from {enc}) and saved as UTF-8.")
    except Exception as e:
        print(f"  [WARN] couldn't replace original file: {e}. Fixed copy located at: {tmp}")
        return False

    # remove possible __pycache__ for this directory
    pycache = p.parent / "__pycache__"
    if pycache.exists():
        try:
            shutil.rmtree(pycache)
            print(f"  removed pycache: {pycache}")
        except Exception as e:
            print(f"  could not remove pycache {pycache}: {e}")

    return True

def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("Repo root:", repo_root)
    targets = find_target_paths(repo_root)
    if not targets:
        print("No DSF target .py files found under external/.../models or src/models/dsf_models.")
        print("Checked paths:")
        print(" -", repo_root / "external" / "DecisionSpikeFormer" / "gym" / "models")
        print(" -", repo_root / "src" / "models" / "dsf_models")
        return 1

    print(f"Found {len(targets)} files to inspect.")
    any_fail = False
    for p in targets:
        ok = repair_file(p)
        if not ok:
            any_fail = True

    if any_fail:
        print("\nOne or more files could not be fully repaired. Please inspect outputs above.")
        return 2
    else:
        print("\nAll target files checked/repaired successfully.")
        print("Now try: python src/run_experiment.py --env CartPole-v1 --model_type snn-dt --epochs 100 --batch_size 64 --lr 3e-4")
        return 0

if __name__ == '__main__':
    sys.exit(main())
