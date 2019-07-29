from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset


@attrs(frozen=True, slots=True)
class LexiconEntry:
    base_form: str = attrib(validator=instance_of(str))
    properties: ImmutableSet["LexiconProperty"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True)
class LexiconProperty:
    name: str = attrib(validator=instance_of(str))

    def __str__(self) -> str:
        return f"+{self.name}"


# certain standard LexiconProperties
NOMINAL = LexiconProperty("nominal")
VERBAL = LexiconProperty("verbal")
