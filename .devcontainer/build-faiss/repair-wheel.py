#!/usr/bin/env python3
"""
Repair faiss wheel to include libfaiss.so dependency.
This script extracts the wheel, copies the required shared libraries,
patches RPATHs, and repackages it.
"""

import sys
import os
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

def repair_wheel(wheel_path, lib_dir):
    """Repair wheel by bundling libfaiss.so"""
    wheel_path = Path(wheel_path)
    lib_dir = Path(lib_dir)
    
    if not wheel_path.exists():
        print(f"Error: Wheel not found: {wheel_path}")
        return False
    
    print(f"Repairing wheel: {wheel_path.name}")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extract_dir = tmpdir / "wheel"
        extract_dir.mkdir()
        
        # Extract wheel
        print("  Extracting wheel...")
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        # Find faiss package directory
        faiss_dir = extract_dir / "faiss"
        if not faiss_dir.exists():
            print(f"Error: faiss directory not found in wheel")
            return False
        
        # Copy required shared libraries
        print("  Copying shared libraries...")
        libs_to_copy = [
            "libfaiss.so",
            "libfaiss_c.so",
        ]
        
        for lib in libs_to_copy:
            src = lib_dir / lib
            if src.exists():
                dst = faiss_dir / lib
                shutil.copy2(src, dst)
                print(f"    ✓ Copied {lib}")
            else:
                print(f"    ⚠ Warning: {lib} not found at {src}")
        
        # Patch RPATH in extension modules to find libraries in same directory
        print("  Patching RPATH in extension modules...")
        so_files = [
            faiss_dir / "_swigfaiss.so",
            faiss_dir / "libfaiss_python_callbacks.so",
            faiss_dir / "_faiss_example_external_module.so",
        ]
        
        for so_file in so_files:
            if so_file.exists():
                try:
                    # Set RPATH to $ORIGIN (same directory as the .so file)
                    subprocess.run(
                        ["patchelf", "--set-rpath", "$ORIGIN", str(so_file)],
                        check=True,
                        capture_output=True
                    )
                    print(f"    ✓ Patched RPATH in {so_file.name}")
                except subprocess.CalledProcessError as e:
                    print(f"    ⚠ Warning: Failed to patch {so_file.name}: {e}")
                except FileNotFoundError:
                    print(f"    ⚠ Warning: patchelf not found, skipping RPATH patching")
                    break
        
        # Update RECORD file
        print("  Updating RECORD...")
        record_path = None
        for path in extract_dir.glob("faiss-*.dist-info/RECORD"):
            record_path = path
            break
        
        if record_path:
            with open(record_path, 'a') as f:
                for lib in libs_to_copy:
                    if (faiss_dir / lib).exists():
                        f.write(f"faiss/{lib},,\n")
        
        # Repackage wheel
        print("  Repackaging wheel...")
        output_wheel = wheel_path.parent / f"{wheel_path.stem}.repaired{wheel_path.suffix}"
        
        with zipfile.ZipFile(output_wheel, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(extract_dir)
                    zf.write(file_path, arcname)
        
        # Replace original wheel
        shutil.move(output_wheel, wheel_path)
        print(f"✓ Repaired wheel: {wheel_path.name}")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: repair-wheel.py <wheel_file> <lib_directory>")
        sys.exit(1)
    
    wheel_file = sys.argv[1]
    lib_dir = sys.argv[2]
    
    if repair_wheel(wheel_file, lib_dir):
        sys.exit(0)
    else:
        sys.exit(1)
