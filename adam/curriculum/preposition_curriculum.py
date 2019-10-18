from itertools import chain
from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.curriculum.phase1_curriculum import (
    Phase1InstanceGroup,
    standard_object,
    phase1_instances,
)
from adam.ontology import IN_REGION, IS_ADDRESSEE, IS_SPEAKER
from adam.ontology.phase1_ontology import (
    BALL,
    BOOK,
    BOX,
    TABLE,
    on,
    GAILA_PHASE_1_ONTOLOGY,
    strictly_above,
    inside,
    WATER,
    JUICE,
    CUP,
    MOM,
    DAD,
)
from adam.ontology.phase1_spatial_relations import Region, PROXIMAL, Direction
from adam.random_utils import RandomChooser
from adam.relation import Relation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    sampled,
    TemplateObjectVariable,
    object_variable,
)

_CHOOSER = RandomChooser.for_seed(0)


def _make_background(
    salient: Iterable[TemplateObjectVariable],
    all_objects: Iterable[TemplateObjectVariable],
) -> Iterable[TemplateObjectVariable]:
    return immutableset(object_ for object_ in all_objects if object_ not in salient)


def _make_on_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def on_templates(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-on-{ground.handle}",
            salient_object_variables=[figure, ground],
            background_object_variables=background,
            asserted_always_relations=[on(figure, ground)],
        )

    return phase1_instances(
        "Preposition Training ON",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            on_templates(
                                figure,
                                ground,
                                _make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                ),
                            ),
                            chooser=_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_beside_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def beside_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
        *,
        is_right: bool,
    ) -> Phase1SituationTemplate:
        direction_str = "right" if is_right else "left"
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-beside-{ground.handle}-{direction_str}",
            salient_object_variables=[figure, ground],
            background_object_variables=background,
            asserted_always_relations=[
                Relation(
                    IN_REGION,
                    figure,
                    Region(
                        ground,
                        distance=PROXIMAL,
                        direction=Direction(
                            positive=is_right,
                            relative_to_axis=HorizontalAxisOfObject(ground, index=0),
                        ),
                    ),
                )
            ],
        )

    return phase1_instances(
        "Preposition Training Beside",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            beside_template(
                                figure,
                                ground,
                                _make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                ),
                                is_right=direction,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=_CHOOSER,
                            max_to_sample=5,
                        )
                        for figure in figures
                        for ground in grounds
                        for direction in immutableset([True, False])
                    ]
                )
            ]
        ),
    )


def _make_under_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0])

    def under_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-under-{ground.handle}",
            salient_object_variables=[figure],
            background_object_variables=background,
            asserted_always_relations=[strictly_above(ground, figure)],
        )

    return phase1_instances(
        "Preposition Training Under",
        chain(
            *[
                sampled(
                    under_template(
                        figure,
                        ground,
                        _make_background(
                            [figure], all_objects=flatten([figures, grounds])
                        ),
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=_CHOOSER,
                    max_to_sample=5,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_over_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def over_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-over-{ground.handle}",
            salient_object_variables=[figure],
            background_object_variables=background,
            asserted_always_relations=[strictly_above(figure, ground)],
        )

    return phase1_instances(
        "Preposition Training Over",
        chain(
            *[
                sampled(
                    over_template(
                        figure,
                        ground,
                        _make_background(
                            [figure], all_objects=flatten([figures, grounds])
                        ),
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=_CHOOSER,
                    max_to_sample=5,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_in_training() -> Phase1InstanceGroup:
    figure_0 = object_variable("water", WATER)
    figure_1 = object_variable("juice", JUICE)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("cup", CUP)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def in_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-in-{ground.handle}",
            salient_object_variables=[figure, ground],
            background_object_variables=background,
            asserted_always_relations=[inside(figure, ground)],
        )

    return phase1_instances(
        "Preposition Training In",
        chain(
            *[
                sampled(
                    in_template(
                        figure,
                        ground,
                        _make_background(
                            [figure, ground], all_objects=flatten([figures, grounds])
                        ),
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=_CHOOSER,
                    max_to_sample=5,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_behind_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("table", TABLE)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", DAD, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def behind_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-behind-{ground.handle}",
            salient_object_variables=[figure, ground],
            background_object_variables=background,
            asserted_always_relations=[
                Relation(
                    IN_REGION,
                    figure,
                    Region(
                        ground,
                        distance=PROXIMAL,
                        direction=Direction(
                            positive=False, relative_to_axis=FacingAddresseeAxis(ground)
                        ),
                    ),
                )
            ],
        )

    return phase1_instances(
        "Preposition Training Behind",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            behind_template(
                                figure,
                                ground,
                                _make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                ),
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=_CHOOSER,
                            max_to_sample=5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_in_front_training() -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("table", TABLE)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", DAD, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    def in_front_template(
        figure: TemplateObjectVariable,
        ground: TemplateObjectVariable,
        background: Iterable[TemplateObjectVariable],
    ) -> Phase1SituationTemplate:
        return Phase1SituationTemplate(
            f"preposition-training-{figure.handle}-behind-{ground.handle}",
            salient_object_variables=[figure, ground],
            background_object_variables=background,
            asserted_always_relations=[
                Relation(
                    IN_REGION,
                    figure,
                    Region(
                        ground,
                        distance=PROXIMAL,
                        direction=Direction(
                            positive=True, relative_to_axis=FacingAddresseeAxis(ground)
                        ),
                    ),
                )
            ],
        )

    return phase1_instances(
        "Preposition Training In Front",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            in_front_template(
                                figure,
                                ground,
                                _make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                ),
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=_CHOOSER,
                            max_to_sample=5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_on_tests() -> Phase1InstanceGroup:
    pass


def _make_beside_tests() -> Phase1InstanceGroup:
    pass


def _make_under_tests() -> Phase1InstanceGroup:
    pass


def _make_over_tests() -> Phase1InstanceGroup:
    pass


def _make_in_tests() -> Phase1InstanceGroup:
    pass


def _make_behind_tests() -> Phase1InstanceGroup:
    pass


def _make_in_front_tests() -> Phase1InstanceGroup:
    pass


PREPOSITIONS_CURRICULUM_TRAINING = [
    _make_on_training(),
    _make_beside_training(),
    _make_under_training(),
    _make_over_training(),
    _make_in_training(),
    _make_behind_training(),
    _make_in_front_training(),
]

PREPOSITIONS_CURRICULUM_TESTING = [
    _make_on_tests(),
    _make_beside_tests(),
    _make_under_tests(),
    _make_over_tests(),
    _make_in_tests(),
    _make_behind_tests(),
    _make_in_front_tests(),
]

PREPOSITIONS_CURRICULUM = flatten([PREPOSITIONS_CURRICULUM_TRAINING])
