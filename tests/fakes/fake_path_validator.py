from src.spira_training.shared.ports.path_validator import PathValidator

class FakePathValidator(PathValidator):
    def validate_path(self, path: str) -> str:
        return path