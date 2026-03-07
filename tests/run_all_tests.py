# tests/run_all_tests.py

"""
run_all_tests.py
----------------
Runs the full test battery and prints a summary.
Usage:
  python tests/run_all_tests.py
"""

import subprocess
import sys

TEST_MODULES = [
    "tests/test_transcript_extractor.py",
    "tests/test_cleaner.py",
    "tests/test_chunker.py",
    "tests/test_live_ingest.py",
    "tests/test_ingest_node.py",
]

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *TEST_MODULES, "-v", "--tb=short"],
        cwd=None,
    )
    sys.exit(result.returncode)