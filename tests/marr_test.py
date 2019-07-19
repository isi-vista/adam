from immutablecollections import immutabledict

from adam.perception.marr import (
    feet_to_meters,
    Cylinder,
    inches_to_meters,
    Marr3dObject,
    AdjunctRelation,
)



def make_test_human_object():
    person_instance_bounding_cylinder = Cylinder(
        # person height
        length_in_meters=feet_to_meters(5.0 + 5.0 / 12.0),
        # person width
        diameter_in_meters=inches_to_meters(15.0),
        # person depth
        # minor_diameter_in_meters=inches_to_meters(9.0),
    )

    arm = Marr3dObject.create_from_bounding_cylinder(
        Cylinder(
            length_in_meters=inches_to_meters(31.0),
            diameter_in_meters=inches_to_meters(3.5),
            # minor_diameter_in_meters=inches_to_meters(3.5),
        )
    )

    left_arm_orientation = AdjunctRelation.Orientation(
        theta_in_degrees=90.0,
        r_relative_to_reference_width=1.0,
        p_relative_to_reference_length=0.9,
        phi_in_degrees=0.0,
        iota_in_degrees=90.0,
    )

    right_arm_orientation = AdjunctRelation.Orientation(
        theta_in_degrees=-90.0,
        r_relative_to_reference_width=1.0,
        p_relative_to_reference_length=0.9,
        phi_in_degrees=0.0,
        iota_in_degrees=90.0,
    )

    leg = Marr3dObject.create_from_bounding_cylinder(
        Cylinder(
            length_in_meters=inches_to_meters(33.0),
            diameter_in_meters=inches_to_meters(6.0),
            # minor_diameter_in_meters=inches_to_meters(5.5),
        )
    )

    left_leg_orientation = AdjunctRelation.Orientation(
        theta_in_degrees=90.0,
        r_relative_to_reference_width=1.0,
        p_relative_to_reference_length=0.0,
        phi_in_degrees=90.0,
        iota_in_degrees=180.0,
    )

    right_leg_orientation = AdjunctRelation.Orientation(
        theta_in_degrees=-90.0,
        r_relative_to_reference_width=1.0,
        p_relative_to_reference_length=0.0,
        phi_in_degrees=90.0,
        iota_in_degrees=180.0,
    )

    torso = Marr3dObject.create_from_bounding_cylinder(
        Cylinder(
            length_in_meters=inches_to_meters(28.0),
            diameter_in_meters=inches_to_meters(15.0),
            # minor_diameter_in_meters=inches_to_meters(9.0),
        )
    )

    head = Marr3dObject.create_from_bounding_cylinder(
        Cylinder(
            # includes neck
            length_in_meters=inches_to_meters(11.5),
            diameter_in_meters=inches_to_meters(8.5),
            # minor_diameter_in_meters=inches_to_meters(8.0),
        )
    )

    head_orientation = (
        AdjunctRelation.Orientation(
            # top of the torso
            p_relative_to_reference_length=1.0,
            # middle of the top of the torso
            r_relative_to_reference_width=0.0,
            # doesn't matter because r = 0
            theta_in_degrees=0.0,
            # vector through head points straight up, just like the primary axis of the torso
            iota_in_degrees=0.0,
            phi_in_degrees=0.0,
        ),
    )

    return Marr3dObject(
        bounding_cylinder=person_instance_bounding_cylinder,
        # its easier to specify human body parts with respect to the torso
        principal_cylinder=torso.bounding_cylinder,
        components=immutabledict(
            [
                (torso, AdjunctRelation.Orientation.create_same_as_reference_cylinder()),
                (head, head_orientation),
                (arm, left_arm_orientation),
                (arm, right_arm_orientation),
                (leg, left_leg_orientation),
                (leg, right_leg_orientation),
            ]
        ),
    )