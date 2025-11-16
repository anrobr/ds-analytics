#!/usr/bin/env python3
"""
Patch faiss setup.py to generate proper platform-specific wheel tags.
The issue is that faiss uses package_data for .so files instead of ext_modules,
which causes setuptools to generate a 'none-any' wheel instead of platform-specific.
"""

import sys

setup_py_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/faiss/faiss/python/setup.py"

with open(setup_py_path, 'r') as f:
    content = f.read()

# Add this import at the top if not present
if "from setuptools.dist import Distribution" not in content:
    content = content.replace(
        "from setuptools import setup",
        "from setuptools import setup\nfrom setuptools.dist import Distribution"
    )

# Add BinaryDistribution class before setup() call
binary_dist_class = '''
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True

'''

if "class BinaryDistribution" not in content:
    content = content.replace(
        "setup(",
        binary_dist_class + "setup(\n    distclass=BinaryDistribution,"
    )

with open(setup_py_path, 'w') as f:
    f.write(content)

print(f"✓ Patched {setup_py_path} to generate platform-specific wheels")
