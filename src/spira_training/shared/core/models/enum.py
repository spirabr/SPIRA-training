from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def has_name(cls, name) -> bool:
        return name in cls._member_names_

    @classmethod
    def has_value(cls, value) -> bool:
        return value in cls._value2member_map_
