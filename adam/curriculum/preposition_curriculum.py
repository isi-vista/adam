from itertools import chain
from typing import Iterable, Sequence, Optional
from immutablecollections import immutableset
from more_itertools import flatten
from adam.language.language_generator import LanguageGenerator
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.language.dependency import LinearizedDependencyTree
from adam.axes import HorizontalAxisOfObject, FacingAddresseeAxis
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    standard_object,
    Phase1InstanceGroup,
    phase1_instances,
    make_noise_objects,
)
from adam.language_specific.english.english_language_generator import (
    USE_ABOVE_BELOW,
    USE_NEAR,
)
from adam.ontology import IS_ADDRESSEE, IS_SPEAKER, THING, OntologyNode
from adam.ontology.phase1_ontology import (
    BALL,
    BOOK,
    BOX,
    TABLE,
    on,
    GAILA_PHASE_1_ONTOLOGY,
    inside,
    WATER,
    JUICE,
    CUP,
    MOM,
    COOKIE,
    CHAIR,
    DAD,
    PERSON,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    bigger_than,
    HAS_SPACE_UNDER,
    IS_BODY_PART,
    HOLLOW,
    near,
    far,
    strictly_under,
    strictly_over,
    LEARNER,
)
from adam.ontology.phase1_spatial_relations import PROXIMAL, Direction, DISTAL
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    sampled,
    TemplateObjectVariable,
    object_variable,
)

BOOL_SET = immutableset([True, False])


# TODO: fix https://github.com/isi-vista/adam/issues/917 which causes us to have to specify that we don't wish to include ME_HACK and YOU_HACK in our curriculum design


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
            near(
                figure,
                ground,
                direction=Direction(
                    positive=is_right,
                    relative_to_axis=HorizontalAxisOfObject(ground, index=0),
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
    is_distal: bool,
    syntax_hints: Iterable[str] = [],
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    # TODO: currently this hack keeps old implementation for English that hasn't solved https://github.com/isi-vista/adam/issues/802
    # and returns new implementation for Chinese that does solve this
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-under-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[
            strictly_under(ground, figure, dist=DISTAL if is_distal else PROXIMAL)
        ],
        constraining_relations=[bigger_than(ground, figure)],
        gazed_objects=[figure],
        syntax_hints=syntax_hints,
    )


def _over_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
    is_distal: bool,
    syntax_hints: Iterable[str] = [],
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    # TODO: currently this hack keeps old implementation for English that hasn't solved https://github.com/isi-vista/adam/issues/802
    # and returns new implementation for Chinese that does solve this
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-over-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[
            strictly_over(figure, ground, dist=DISTAL if is_distal else PROXIMAL)
        ],
        gazed_objects=[figure],
        syntax_hints=syntax_hints,
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
    is_near: bool,
    speaker_root_node: OntologyNode = PERSON,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    direction = Direction(positive=False, relative_to_axis=FacingAddresseeAxis(ground))
    speaker = standard_object("speaker", speaker_root_node, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])
    computed_background = [speaker, addressee]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-behind-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=computed_background,
        asserted_always_relations=[
            near(figure, ground, direction=direction)
            if is_near
            else far(figure, ground, direction=direction)
        ],
        gazed_objects=[figure],
    )


def _in_front_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
    is_near: bool,
    speaker_root_node: OntologyNode = PERSON,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    direction = Direction(positive=True, relative_to_axis=FacingAddresseeAxis(ground))
    speaker = standard_object("speaker", speaker_root_node, added_properties=[IS_SPEAKER])
    addressee = standard_object("addressee", LEARNER, added_properties=[IS_ADDRESSEE])
    computed_background = [speaker, addressee]
    computed_background.extend(background)
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-behind-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=computed_background,
        asserted_always_relations=[
            near(figure, ground, direction=direction)
            if is_near
            else far(figure, ground, direction=direction)
        ],
        gazed_objects=[figure],
    )


def _near_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-near-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[near(figure, ground)],
        gazed_objects=[figure],
        syntax_hints=[USE_NEAR],
    )


def _far_template(
    figure: TemplateObjectVariable,
    ground: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_training: bool,
) -> Phase1SituationTemplate:
    handle = "training" if is_training else "testing"
    return Phase1SituationTemplate(
        f"preposition-{handle}-{figure.handle}-far-{ground.handle}",
        salient_object_variables=[figure, ground],
        background_object_variables=background,
        asserted_always_relations=[far(figure, ground)],
        gazed_objects=[figure],
    )


def _make_on_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("mom", MOM)
    ground_0 = standard_object("chair", CHAIR)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1, figure_2])
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
                                make_noise_objects(noise_objects),
                                is_training=True,
                            ),
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_beside_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("mom", MOM)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    ground_2 = standard_object("dad", DAD)

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1, ground_2])

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
                                make_noise_objects(noise_objects),
                                is_right=direction,
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for direction in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_under_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("mom", MOM)
    ground_0 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0])

    return phase1_instances(
        "Preposition Training Under",
        chain(
            *[
                sampled(
                    _under_template(
                        figure,
                        ground,
                        make_noise_objects(noise_objects),
                        is_training=True,
                        is_distal=distance,
                        syntax_hints=[USE_ABOVE_BELOW] if use_above_below else [],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
                for distance in BOOL_SET
                for use_above_below in BOOL_SET
            ]
        ),
        language_generator=language_generator,
    )


def _make_over_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("mom", MOM)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Training Over",
        chain(
            *[
                sampled(
                    _over_template(
                        figure,
                        ground,
                        make_noise_objects(noise_objects),
                        is_training=True,
                        is_distal=distance,
                        syntax_hints=[USE_ABOVE_BELOW] if use_above_below else [],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
                for distance in BOOL_SET
                for use_above_below in BOOL_SET
            ]
        ),
        language_generator=language_generator,
    )


def _make_in_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
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
                        make_noise_objects(noise_objects),
                        is_training=True,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
        language_generator=language_generator,
    )


def _make_behind_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("dad", DAD)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    ground_2 = standard_object(
        "person", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1, ground_2])

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
                                make_noise_objects(noise_objects),
                                is_training=True,
                                is_near=close,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for close in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_in_front_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("dad", DAD)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    ground_2 = standard_object(
        "person", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1, ground_2])

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
                                make_noise_objects(noise_objects),
                                is_training=True,
                                is_near=close,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for close in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_near_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("dad", DAD)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    ground_2 = standard_object(
        "person", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1, ground_2])

    return phase1_instances(
        "Preposition Training Near",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _near_template(
                                figure,
                                ground,
                                make_noise_objects(noise_objects),
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_far_training(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object("ball", BALL)
    figure_1 = standard_object("book", BOOK)
    figure_2 = standard_object("dad", DAD)
    ground_0 = standard_object("cookie", COOKIE)
    ground_1 = standard_object("table", TABLE)
    ground_2 = standard_object(
        "person", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    figures = immutableset([figure_0, figure_1, figure_2])
    grounds = immutableset([ground_0, ground_1, ground_2])

    return phase1_instances(
        "Preposition Training Far",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _far_template(
                                figure,
                                ground,
                                make_noise_objects(noise_objects),
                                is_training=True,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_on_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
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
                                make_noise_objects(noise_objects),
                                is_training=False,
                            ),
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_beside_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

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
                                make_noise_objects(noise_objects),
                                is_right=direction,
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for direction in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_under_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
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
                        make_noise_objects(noise_objects),
                        is_training=False,
                        is_distal=distance,
                        syntax_hints=[USE_ABOVE_BELOW] if use_above_below else [],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
                for distance in BOOL_SET
                for use_above_below in BOOL_SET
            ]
        ),
        language_generator=language_generator,
    )


def _make_over_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

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
                        make_noise_objects(noise_objects),
                        is_training=False,
                        is_distal=distance,
                        syntax_hints=[USE_ABOVE_BELOW] if use_above_below else [],
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
                for distance in BOOL_SET
                for use_above_below in BOOL_SET
            ]
        ),
        language_generator=language_generator,
    )


def _make_in_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = object_variable(
        "figure_0", THING, banned_properties=[IS_BODY_PART, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[IS_BODY_PART, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0",
        THING,
        required_properties=[HOLLOW],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    ground_1 = standard_object(
        "ground_1",
        THING,
        required_properties=[HOLLOW],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )

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
                        make_noise_objects(noise_objects),
                        is_training=False,
                    ),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    max_to_sample=num_samples if num_samples else 5,
                )
                for figure in figures
                for ground in grounds
            ]
        ),
        language_generator=language_generator,
    )


def _make_behind_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

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
                                make_noise_objects(noise_objects),
                                is_training=False,
                                is_near=close,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for close in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_in_front_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

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
                                make_noise_objects(noise_objects),
                                is_training=False,
                                is_near=close,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                        for close in BOOL_SET
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_near_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing Near",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _near_template(
                                figure,
                                ground,
                                make_noise_objects(noise_objects),
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def _make_far_tests(
    num_samples: Optional[int],
    noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Phase1InstanceGroup:
    figure_0 = standard_object(
        "figure_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    figure_1 = standard_object(
        "figure_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_0 = standard_object(
        "ground_0", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )
    ground_1 = standard_object(
        "ground_1", THING, banned_properties=[HOLLOW, IS_SPEAKER, IS_ADDRESSEE]
    )

    figures = immutableset([figure_0, figure_1])
    grounds = immutableset([ground_0, ground_1])

    return phase1_instances(
        "Preposition Testing Far",
        chain(
            *[
                flatten(
                    [
                        sampled(
                            _far_template(
                                figure,
                                ground,
                                make_noise_objects(noise_objects),
                                is_training=False,
                            ),
                            ontology=GAILA_PHASE_1_ONTOLOGY,
                            chooser=PHASE1_CHOOSER_FACTORY(),
                            max_to_sample=num_samples if num_samples else 5,
                        )
                        for figure in figures
                        for ground in grounds
                    ]
                )
            ]
        ),
        language_generator=language_generator,
    )


def make_prepositions_curriculum_training(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_on_training(num_samples, num_noise_objects, language_generator),
        _make_beside_training(num_samples, num_noise_objects, language_generator),
        _make_under_training(num_samples, num_noise_objects, language_generator),
        _make_over_training(num_samples, num_noise_objects, language_generator),
        _make_in_training(num_samples, num_noise_objects, language_generator),
        _make_behind_training(num_samples, num_noise_objects, language_generator),
        _make_in_front_training(num_samples, num_noise_objects, language_generator),
        _make_near_training(num_samples, num_noise_objects, language_generator),
        _make_far_training(num_samples, num_noise_objects, language_generator),
    ]


def make_prepositions_curriculum_testing(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return [
        _make_on_tests(num_samples, num_noise_objects, language_generator),
        _make_beside_tests(num_samples, num_noise_objects, language_generator),
        _make_under_tests(num_samples, num_noise_objects, language_generator),
        _make_over_tests(num_samples, num_noise_objects, language_generator),
        _make_in_tests(num_samples, num_noise_objects, language_generator),
        _make_behind_tests(num_samples, num_noise_objects, language_generator),
        _make_in_front_tests(num_samples, num_noise_objects, language_generator),
        _make_near_tests(num_samples, num_noise_objects, language_generator),
        _make_far_tests(num_samples, num_noise_objects, language_generator),
    ]


def make_prepositions_curriculum(
    num_samples: Optional[int],
    num_noise_objects: Optional[int],
    language_generator: LanguageGenerator[
        HighLevelSemanticsSituation, LinearizedDependencyTree
    ],
) -> Sequence[Phase1InstanceGroup]:
    return flatten(
        [
            make_prepositions_curriculum_training(
                num_samples, num_noise_objects, language_generator
            ),
            make_prepositions_curriculum_testing(
                num_samples, num_noise_objects, language_generator
            ),
        ]
    )
