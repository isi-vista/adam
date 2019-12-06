from itertools import chain
from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER,
    make_background,
    standard_object,
    Phase1InstanceGroup,
    phase1_instances,
)
from adam.ontology import IN_REGION, IS_ADDRESSEE, IS_SPEAKER, THING
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
    COOKIE,
    CHAIR,
    LEARNER,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    bigger_than,
    HAS_SPACE_UNDER,
    IS_BODY_PART,
    HOLLOW,
)
from adam.ontology.phase1_spatial_relations import Region, PROXIMAL, Direction
from adam.relation import Relation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    sampled,
    TemplateObjectVariable,
    object_variable,
)


def _on_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-on-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[on(figure, ground)],
        gazed_objects=[figure],
    )


def _beside_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
    is_training: bool,
) -> Phase1SituationTemplate:
    direction_str = "right" if is_right else "left"
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-beside-{ground.handle}-{direction_str}",
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
        gazed_objects=[figure],
    )


def _under_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-under-{ground.handle}",
        salient_object_variables=[ground],
        background_object_variables=background,
        asserted_always_relations=[strictly_above(ground, figure)],
        constraining_relations=[bigger_than(ground, figure)],
        gazed_objects=[figure],
    )


def _over_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-over-{ground.handle}",
        salient_object_variables=[ground],
        background_object_variables=background,
        asserted_always_relations=[strictly_above(figure, ground)],
        gazed_objects=[figure],
    )


def _in_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-in-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[inside(figure, ground)],
        gazed_objects=[figure],
    )


def _behind_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-behind-{ground.handle}",
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
        gazed_objects=[figure],
    )


def _in_front_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-behind-{ground.handle}",
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
        gazed_objects=[figure],
    )


def _make_on_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("chair", CHAIR)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training On",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _on_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                )
                                if noise_objects
                                else immutableset(),
                                is_training=True,
                            ),
                            chooser=PHASE1_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_beside_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training Beside",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _beside_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                )
                                if noise_objects
                                else immutableset(),
                                is_right=direction,
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                        for direction in immutableset([True, False])
                    ]
                )
            ]
        ),
    )


def _make_under_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0])

    return phase1_instances(
        "Preposition Training Under",
        chain(
            *[
                sampled(
                    _under_template(
                        figure,
                        ground,
                        make_background([figure], all_objects=flatten([figures, grounds]))
                        if noise_objects
                        else immutableset(),
                        is_training=True,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_over_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training Over",
        chain(
            *[
                sampled(
                    _over_template(
                        figure,
                        ground,
                        make_background([figure], all_objects=flatten([figures, grounds]))
                        if noise_objects
                        else immutableset(),
                        is_training=True,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_in_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = object_variable("water", WATER)
    figure_1 = object_variable("juice", JUICE)
    ground_0 = standard_object("box", BOX)
    ground_1 = standard_object("cup", CUP)

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training In",
        chain(
            *[
                sampled(
                    _in_template(
                        figure,
                        ground,
                        make_background(
                            [figure, ground], all_objects=flatten([figures, grounds])
                        )
                        if noise_objects
                        else immutableset(),
                        is_training=True,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_behind_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training Behind",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _behind_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                )
                                if noise_objects
                                else immutableset([speaker, addressee]),
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_in_front_training(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training In Front",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _in_front_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                )
                                if noise_objects
                                else immutableset([speaker, addressee]),
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_on_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object(
        "ground_0",
        THING,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
        banned_properties=[HOLLOW],
    )
    ground_1 = standard_object(
        "ground_1",
        THING,
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM],
        banned_properties=[HOLLOW],
    )

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing On",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _on_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                )
                                if noise_objects
                                else immutableset(),
                                is_training=False,
                            ),
                            chooser=PHASE1_CHOOSER,
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_beside_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object("ground_0", THING, banned_properties=[HOLLOW])
    ground_1 = standard_object("ground_1", THING, banned_properties=[HOLLOW])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing Beside",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _beside_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten([figures, grounds]),
                                )
                                if noise_objects
                                else immutableset(),
                                is_right=direction,
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                        for direction in immutableset([True, False])
                    ]
                )
            ]
        ),
    )


def _make_under_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object(
        "ground_0",
        THING,
        required_properties=[HAS_SPACE_UNDER],
        banned_properties=[HOLLOW],
    )

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0])

    return phase1_instances(
        "Preposition Testing Under",
        chain(
            *[
                sampled(
                    _under_template(
                        figure,
                        ground,
                        make_background([figure], all_objects=flatten([figures, grounds]))
                        if noise_objects
                        else immutableset(),
                        is_training=False,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_over_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object("ground_0", THING, banned_properties=[HOLLOW])
    ground_1 = standard_object("ground_1", THING, banned_properties=[HOLLOW])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing Over",
        chain(
            *[
                sampled(
                    _over_template(
                        figure,
                        ground,
                        make_background([figure], all_objects=flatten([figures, grounds]))
                        if noise_objects
                        else immutableset(),
                        is_training=False,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_in_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = object_variable("figure_0", THING, banned_properties=[IS_BODY_PART])
    figure_1 = standard_object("figure_1", THING, banned_properties=[IS_BODY_PART])
    ground_0 = standard_object("ground_0", THING, required_properties=[HOLLOW])
    ground_1 = standard_object("ground_1", THING, required_properties=[HOLLOW])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing In",
        chain(
            *[
                sampled(
                    _in_template(
                        figure,
                        ground,
                        make_background(
                            [figure, ground], all_objects=flatten([figures, grounds])
                        )
                        if noise_objects
                        else immutableset(),
                        is_training=False,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER,
                    max_to_sample=num_samples,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
    )


def _make_behind_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object("ground_0", THING, banned_properties=[HOLLOW])
    ground_1 = standard_object("ground_1", THING, banned_properties=[HOLLOW])

    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing Behind",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _behind_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                )
                                if noise_objects
                                else immutableset([speaker, addressee]),
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def _make_in_front_tests(
    num_samples: int = 5, *, noise_objects: bool = True
) -> Phase1InstanceGroup:
    figure_0 = standard_object("figure_0", THING, banned_properties=[HOLLOW])
    figure_1 = standard_object("figure_1", THING, banned_properties=[HOLLOW])
    ground_0 = standard_object("ground_0", THING, banned_properties=[HOLLOW])
    ground_1 = standard_object("ground_1", THING, banned_properties=[HOLLOW])

    speaker = standard_object("speaker", MOM, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing In Front",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _in_front_template(
                                figure,
                                ground,
                                make_background(
                                    [figure, ground],
                                    all_objects=flatten(
                                        [figures, grounds, [speaker, addressee]]
                                    ),
                                )
                                if noise_objects
                                else immutableset([speaker, addressee]),
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER,
                            max_to_sample=num_samples,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
    )


def make_prepositions_curriculum_training(
    num_samples: int = 5, *, noise_objects: bool = True
):
    return [
        _make_on_training(num_samples, noise_objects=noise_objects),
        _make_beside_training(num_samples, noise_objects=noise_objects),
        _make_under_training(num_samples, noise_objects=noise_objects),
        _make_over_training(num_samples, noise_objects=noise_objects),
        _make_in_training(num_samples, noise_objects=noise_objects),
        _make_behind_training(num_samples, noise_objects=noise_objects),
        _make_in_front_training(num_samples, noise_objects=noise_objects),
    ]


def make_prepositions_curriculum_testing(
    num_samples: int = 5, *, noise_objects: bool = True
):
    return [
        _make_on_tests(num_samples, noise_objects=noise_objects),
        _make_beside_tests(num_samples, noise_objects=noise_objects),
        _make_under_tests(num_samples, noise_objects=noise_objects),
        _make_over_tests(num_samples, noise_objects=noise_objects),
        _make_in_tests(num_samples, noise_objects=noise_objects),
        _make_behind_tests(num_samples, noise_objects=noise_objects),
        _make_in_front_tests(num_samples, noise_objects=noise_objects),
    ]


def make_prepositions_curriculum(num_samples: int = 5, *, noise_objects: bool = True):
    return flatten(
        [
            make_prepositions_curriculum_training(
                num_samples, noise_objects=noise_objects
            ),
            make_prepositions_curriculum_testing(
                num_samples, noise_objects=noise_objects
            ),
        ]
    )
