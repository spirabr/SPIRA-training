from pydantic import BaseModel


class BaseTestModel(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }
