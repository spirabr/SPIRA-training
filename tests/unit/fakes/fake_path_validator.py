from pathlib import Path

from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.ports.path_validator import PathValidator

class FakePathValidator(PathValidator):
    def validate_path(self, path: Path | str) -> ValidPath:
        if isinstance(path, str):
            path = Path(path)
        return ValidPath(path=path)