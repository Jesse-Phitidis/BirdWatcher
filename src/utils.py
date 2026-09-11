import sys
from pathlib import Path

def get_asset_path(relative_path: str) -> Path:
    """Get the absolute path to a resource, whether in dev or PyInstaller"""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)
    else:
        # Normal dev mode: go up one directory from src/ to the project root
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path
