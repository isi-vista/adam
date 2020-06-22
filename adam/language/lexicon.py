"""
Data structures for human language words.

These are not used by the `LanguageLearner`,
but rather for generating the linguistic descriptions for situations.
"""
from typing import Optional

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.language.dependency import MorphosyntacticProperty, PartOfSpeechTag


@attrs(frozen=True, slots=True)
class LexiconEntry:
    base_form: str = attrib(validator=instance_of(str))
    """
    The base linguistic form for a `LexiconEntry`.

    What form is chosen as the base form varies by part-of-speech and from language to language.

    For example, in English, the base form for nouns might be the singular, while the base form
    for verbs might be the present tense.
    """
    part_of_speech: PartOfSpeechTag = attrib(validator=instance_of(PartOfSpeechTag))
    properties: ImmutableSet["LexiconProperty"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    intrinsic_morphosyntactic_properties: ImmutableSet[MorphosyntacticProperty] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    counting_classifier: Optional[str] = attrib(default=None, kw_only=True)
    plural_form: Optional[str] = attrib(default=None, kw_only=True)
    verb_form_sg3_prs: Optional[str] = attrib(default=None, kw_only=True)


@attrs(frozen=True, slots=True)
class LexiconProperty:
    """
    A linguistic property that a `LexiconEntry` may possess.

    For example, *singular*, *active*, etc.
    """

    name: str = attrib(validator=instance_of(str))

    def __str__(self) -> str:
        return f"+{self.name}"
