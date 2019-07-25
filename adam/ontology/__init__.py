from attr import attrs, attrib
from attr.validators import instance_of


@attrs(frozen=True, slots=True)
class OntologyNode:
    handle: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True)
class OntologyProperty:
    handle: str = attrib(validator=instance_of(str))
