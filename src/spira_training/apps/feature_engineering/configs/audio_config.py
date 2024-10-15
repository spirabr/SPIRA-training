from pathlib import Path

from pydantic import BaseModel

class DatasetPaths(BaseModel):
    patients_csv: Path
    controls_csv: Path
    noises_csv: Path

class AudioConfig(BaseModel):
    dataset_paths: DatasetPaths
    normalize: bool