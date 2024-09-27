from pathlib import Path
from pydantic import BaseModel

class ValidPath(BaseModel):
    path: Path

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)