"""Ensure the project root is importable during tests (api., src., db.)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
