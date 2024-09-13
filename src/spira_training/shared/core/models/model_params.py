from typing_extensions import TypeVar
from pydantic import BaseModel


class ModelParams(BaseModel): ...


ModelParamsT = TypeVar("ModelParamsT", bound=ModelParams)
