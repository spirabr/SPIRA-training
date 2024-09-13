from pathlib import Path
from typing import Any
from pydantic import BaseModel, model_serializer, model_validator

class ValidPath(BaseModel):
    path: Path

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    @classmethod
    def from_str(cls, data: str):
        return ValidPath.model_construct(path=create_valid_path(data))

    @classmethod
    def from_list(cls, data: list[str]) -> list['ValidPath']:
        return [cls.from_str(item) for item in data]

    @classmethod
    @model_validator(mode="before")
    def read(cls, data: Any) -> dict[str, Path]:
        if isinstance(data, str):
            return {"path": create_valid_path(data)}
        if isinstance(data, dict):
            return data
        raise RuntimeError("Path should be a string.")

    @model_serializer()
    def write(self) -> Path:
        return self.path

def create_valid_path(unvalidated_path: str) -> Path:
    path = Path(unvalidated_path)
    check_file_exists_or_raise(path)

    return path

def check_file_exists_or_raise(path: Path):
    if not path.is_file():
        raise FileExistsError(f"File {path} does not exist.")