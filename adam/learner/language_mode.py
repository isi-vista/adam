from enum import Enum, auto
from typing import Mapping

from immutablecollections import immutabledict


class LanguageMode(Enum):
    ENGLISH = auto()
    CHINESE = auto()
    BILINGUAL = auto()


LANGUAGE_MODE_TO_NAME: Mapping[LanguageMode, str] = immutabledict(
    {
        LanguageMode.ENGLISH: "english",
        LanguageMode.CHINESE: "chinese",
        LanguageMode.BILINGUAL: "bilingual",
    }
)
