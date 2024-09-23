from typing import Literal
from pydantic import BaseModel
from src.spira_training.shared.core.models.enum import BaseEnum
from src.spira_training.shared.core.models.loss import Loss


class BaseEvent(BaseModel): ...


class EventTypeEnum(BaseEnum):
    TRAIN_LOSS = "train_loss"


class TrainLossEvent(BaseEvent):
    loss: Loss
    type: Literal[EventTypeEnum.TRAIN_LOSS] = EventTypeEnum.TRAIN_LOSS


Event = TrainLossEvent
