from typing import List
from pydantic import BaseModel


class Loss(BaseModel):
    value: List[float] | float
