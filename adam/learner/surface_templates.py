"""
Representations of template-with-slots-like patterns over token strings.
"""
from attr import attrs, attrib
from attr.validators import instance_of


@attrs(frozen=True, slots=True)
class SurfaceTemplateVariable:
    """
    A variable portion of a a `SurfaceTemplate`
    """

    name: str = attrib(validator=instance_of(str))
