from pathlib import Path as PathLibPath
from typing import NewType

__all__ = "Path"


Path = NewType("Path", PathLibPath)
