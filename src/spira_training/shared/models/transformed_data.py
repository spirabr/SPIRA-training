from enum import Enum


class Label:
    patient = 1
    control = 0


class TransformedData:
    def __init__(self, features: List[str], label: Label):
        self.features = features
        self.label = label
