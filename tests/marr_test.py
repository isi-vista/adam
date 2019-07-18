from adam.perception.marr import feet_to_meters, Cylinder, inches_to_meters, Marr3dObject


def test_person():
    # first we construct a model for people in general

    # then we a particular "stick-person" instance
    person_instance_bounding_cylinder = Cylinder(
        # person height
        length_in_meters=feet_to_meters(5.0 + 5.0 / 12.0),
        # person width
        major_diameter_in_meters=inches_to_meters(15.0),
        # person depth
        minor_diameter_in_meters=inches_to_meters(9.0),
    )

    arm_instance_bounding_cylinder = Cylinder(
        length_in_meters=inches_to_meters(31.0),
        major_diameter_in_meters=inches_to_meters(3.5),
        minor_diameter_in_meters=inches_to_meters(3.5),
    )

    leg_instance_bounding_cylinder = Cylinder(
        length_in_meters=inches_to_meters(33.0),
        major_diameter_in_meters=inches_to_meters(6.0),
        minor_diameter_in_meters=inches_to_meters(5.5),
    )

    torso_instance_bounding_cylinder = Cylinder(
        length_in_meters=inches_to_meters(28.0),
        major_diameter_in_meters=inches_to_meters(15.0),
        minor_diameter_in_meters=inches_to_meters(9.0),
    )

    head_instance_bounding_cylinder = Cylinder(
        # includes neck
        length_in_meters=inches_to_meters(11.5),
        major_diameter_in_meters=inches_to_meters(8.5),
        minor_diameter_in_meters=inches_to_meters(8.0),
    )

    person_instance = Marr3dObject(
        bounding_cylinder=person_instance_bounding_cylinder,
        major_axis=None,
        components=[],
    )

    # finally we check that the person instance is compatible with the person model
