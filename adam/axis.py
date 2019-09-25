from attr import attrs, attrib, evolve
from attr.validators import instance_of

from adam.utilities import _sign


@attrs(frozen=True, slots=True, repr=False, cmp=False)
class GeonAxis:
    debug_name: str = attrib(validator=instance_of(str))
    curved: bool = attrib(validator=instance_of(bool), default=False)
    directed: bool = attrib(validator=instance_of(bool), default=True)
    aligned_to_gravitational = attrib(validator=instance_of(bool), default=False)

    def copy(self) -> "GeonAxis":
        return evolve(self)

    def __repr__(self) -> str:
        return (
            f"{self.debug_name}"
            f"[{_sign(self.curved)}curved, "
            f"{_sign(self.directed)}directed, "
            f"{_sign(self.curved)}aligned_to_gravity]"
        )
