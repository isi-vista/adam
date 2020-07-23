from adam.utilities import sign
from attr import attrib, attrs, evolve
from attr.validators import instance_of


@attrs(frozen=True, slots=True, repr=False, eq=False)
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
            f"[{sign(self.curved)}curved, "
            f"{sign(self.directed)}directed, "
            f"{sign(self.aligned_to_gravitational)}aligned_to_gravity]"
        )
