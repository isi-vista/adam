import enum
from immutablecollections import immutableset

OBJECT_NAMES_TO_EXCLUDE = immutableset(["the ground", "learner"])


class Shape(enum.Enum):
    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    OVALISH = "OVALISH"
    RECTANGULAR = "RECTANGULAR"
    IRREGULAR = "IRREGULAR"
