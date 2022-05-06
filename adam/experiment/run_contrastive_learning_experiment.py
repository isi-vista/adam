"""
Script to implement the contrastive learning experiment.

For now this only supports contrastive object learning with a subset object learner.
"""
from collections import defaultdict
import logging
from itertools import product
from random import Random
from typing import (
    Optional,
    Mapping,
    MutableSequence,
    Tuple,
    Sequence,
    Iterable,
    TypeVar,
    cast,
    Generic,
)

from attr import evolve, attrs, attrib
from attr.validators import instance_of
from immutablecollections import immutableset
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum import ExplicitWithSituationInstanceGroup, InstanceGroup
from adam.experiment import execute_experiment, Experiment
from adam.experiment.experiment_utils import ONTOLOGY_STR_TO_ONTOLOGY
from adam.experiment.log_experiment import experiment_from_params
from adam.experiment.observer import YAMLLogger
from adam.language import LinguisticDescriptionT, LinguisticDescription
from adam.learner import (
    LanguagePerceptionSemanticAlignment,
    ComposableLearner,
    LanguageConceptAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.contrastive_learner import (
    TeachingContrastiveObjectLearner,
    LanguagePerceptionSemanticContrast,
)
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import SubsetObjectLearner
from adam.perception import (
    PerceptionT,
    PerceptualRepresentation,
    PerceptualRepresentationFrame,
)
from adam.perception.perception_graph import GraphLogger, PerceptionGraph
from adam.situation import SituationT, Situation

SYMBOLIC = "symbolic"
SIMULATED = "simulated"
T = TypeVar("T")  # pylint:disable=invalid-name


def contrastive_learning_entry_point(params: Parameters) -> None:
    if (
        params.namespace("object_learner")
        and params.string("object_learner.learner_type") != "subset"
    ):
        raise NotImplementedError(
            "Contrastive learning only implemented for *subset* object learners."
        )
    if params.string("attribute_learner.learner_type") != "none":
        raise NotImplementedError("Contrastive learning not implemented for attributes.")
    if params.string("relation_learner.learner_type") != "none":
        raise NotImplementedError("Contrastive learning not implemented for relations.")
    if params.string("action_learner.learner_type") != "none":
        raise NotImplementedError("Contrastive learning not implemented for actions.")

    debug_perception_log_dir = params.optional_creatable_directory(
        "debug_perception_log_dir"
    )
    perception_graph_logger: Optional[GraphLogger]
    if debug_perception_log_dir:
        logging.info(
            "Debug perception graphs will be written to %s", debug_perception_log_dir
        )
        perception_graph_logger = GraphLogger(
            debug_perception_log_dir, enable_graph_rendering=True
        )
    else:
        perception_graph_logger = None

    base_experiment: Experiment[
        Situation, LinguisticDescription, PerceptualRepresentationFrame
    ] = experiment_from_params(params)
    logging.info("Setting up curriculum.")
    contrastive_pairs: Sequence[
        Tuple[
            Tuple[
                Optional[Situation],
                LinguisticDescription,
                PerceptualRepresentation[PerceptualRepresentationFrame],
            ],
            Tuple[
                Optional[Situation],
                LinguisticDescription,
                PerceptualRepresentation[PerceptualRepresentationFrame],
            ],
        ]
    ]
    apprentice_train_instance_groups, contrastive_pairs = make_curriculum(
        [
            instance
            for group in base_experiment.training_stages
            for instance in group.instances()
        ],
        random_seed=params.integer("curriculum_creation_seed"),
        max_contrastive_samples_per_concept_pair=params.integer(
            "max_contrastive_samples_per_concept_pair"
        ),
    )

    experiment = evolve(base_experiment, training_stages=apprentice_train_instance_groups)
    logging.info("Training apprentice.")
    # We no longer support any non-integrated learners, but that hasn't made it into the type system
    # yet, so we need a cast here to tell MyPy we know what we're doing.
    learner = cast(
        IntegratedTemplateLearner[PerceptualRepresentationFrame, LinguisticDescription],
        execute_experiment(
            experiment=experiment,
            log_path=params.optional_creatable_directory("hypothesis_log_dir"),
            log_hypotheses_every_n_examples=params.integer(
                "log_hypothesis_every_n_steps", default=250
            ),
            log_learner_state=params.boolean("log_learner_state", default=True),
            learner_logging_path=params.optional_creatable_directory(
                "experiment_group_dir"
            ),
            starting_point=params.integer("starting_point", default=0),
            point_to_log=params.integer("point_to_log", default=0),
            load_learner_state=params.optional_existing_file("learner_state_path"),
            resume_from_latest_logged_state=params.boolean(
                "resume_from_latest_logged_state", default=False
            ),
            debug_learner_pickling=params.boolean(
                "debug_learner_pickling", default=False
            ),
            perception_graph_logger=perception_graph_logger,
        ),
    )
    learner.log_hypotheses(
        params.creatable_directory("before_contrastive_hypothesis_log_dir")
    )
    logging.info("Starting contrastive learning.")
    object_learner = cast(SubsetObjectLearner, learner.object_learner)
    ontology, _objects, _perception_gen = ONTOLOGY_STR_TO_ONTOLOGY[
        params.string(
            "contrastive_object_learner.ontology",
            valid_options=ONTOLOGY_STR_TO_ONTOLOGY.keys(),
            default="phase2",
        )
    ]
    contrastive_learner = TeachingContrastiveObjectLearner(
        apprentice=object_learner, ontology=ontology
    )
    contrastive_post_observer: Optional[
        YAMLLogger[Situation, LinguisticDescription, PerceptualRepresentationFrame]
    ] = YAMLLogger.from_params(
        name="contrastive_post_observer",
        params=params.namespace("contrastive_post_observer"),
    )
    if contrastive_post_observer is None:
        logging.warning(
            "No contrastive post-observer specified, so experiment will produce no "
            "post-observation output. Only the final hypotheses will be logged. (This is probably "
            "a mistake.)"
        )
    for (situation1, description1, perception1), (
        situation2,
        description2,
        perception2,
    ) in contrastive_pairs:
        contrastive_learner.learn_from(
            make_contrast(
                (
                    AlignableExample(
                        description1, learner.extract_perception_graph(perception1)
                    ),
                    AlignableExample(
                        description2, learner.extract_perception_graph(perception2)
                    ),
                ),
                learners=[],
            )
        )
        if contrastive_post_observer is not None:
            contrastive_post_observer.observe(
                situation1, description1, perception1, learner.describe(perception1)
            )
            contrastive_post_observer.observe(
                situation2, description2, perception2, learner.describe(perception2)
            )
    learner.log_hypotheses(
        params.creatable_directory("after_contrastive_hypothesis_log_dir")
    )
    contrastive_learner.log_hypotheses(
        params.creatable_directory("after_contrastive_hypothesis_log_dir")
    )
    logging.info("Contrastive learning finished.")


@attrs(frozen=True, slots=True)
class AlignableExample(Generic[LinguisticDescriptionT]):
    """
    An example that can be processed by some composable learners to produce a
    `LanguagePerceptionSemanticAlignment`.
    """

    # Mypy thinks these types are incompatible, but in fact the type var has LinguisticDescription
    # as its bound.
    linguistic_description: LinguisticDescriptionT = attrib(validator=instance_of(LinguisticDescription))  # type: ignore
    perception_graph: PerceptionGraph = attrib(validator=instance_of(PerceptionGraph))


def make_contrast(
    pair: Tuple[
        AlignableExample[LinguisticDescriptionT], AlignableExample[LinguisticDescriptionT]
    ],
    *,
    learners: Sequence[ComposableLearner],
) -> LanguagePerceptionSemanticContrast:
    """
    Create a `LanguagePerceptionSemanticContrast` by processing the pair with the learners.
    """
    # Many of these samples may be repeated. It's probably a good idea to process each sample
    # just once if we can do that. For now with only three concepts I doubt it'll be a major
    # problem.
    return LanguagePerceptionSemanticContrast(
        process_with_learners(
            pair[0].linguistic_description, pair[0].perception_graph, learners
        ),
        process_with_learners(
            pair[1].linguistic_description, pair[1].perception_graph, learners
        ),
    )


def process_with_learners(
    linguistic_description: LinguisticDescriptionT,
    perception_graph: PerceptionGraph,
    learners: Sequence[ComposableLearner],
) -> LanguagePerceptionSemanticAlignment:
    """
    Process an instance using the given sub-learners, producing their alignment for the situation.
    """
    result = LanguagePerceptionSemanticAlignment(
        language_concept_alignment=LanguageConceptAlignment.create_unaligned(
            language=linguistic_description
        ),
        perception_semantic_alignment=PerceptionSemanticAlignment(
            perception_graph=perception_graph,
            semantic_nodes=immutableset(),
        ),
    )
    for learner in learners:
        result = learner.enrich_during_learning(result)
    return result


def make_curriculum(
    instances: Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
    *,
    random_seed: int,
    max_contrastive_samples_per_concept_pair: int,
) -> Tuple[
    Sequence[InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]],
    Sequence[
        Tuple[
            Tuple[
                Optional[SituationT],
                LinguisticDescriptionT,
                PerceptualRepresentation[PerceptionT],
            ],
            Tuple[
                Optional[SituationT],
                LinguisticDescriptionT,
                PerceptualRepresentation[PerceptionT],
            ],
        ]
    ],
]:
    """
    Create a curriculum of (1) instances used to train the apprentice and (2) contrastive pairs for
    the contrastive learner.
    """
    rng = Random(random_seed)
    apprentice_train_instances, contrastive_instances = split_instances(
        instances, rng=rng
    )
    train_instance_groups: Sequence[
        InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = [
        ExplicitWithSituationInstanceGroup(
            name=description.as_token_string(), instances=instances  # type: ignore
        )
        for description, instances in group_by_description(
            apprentice_train_instances
        ).items()
    ]
    contrastive_pairs = make_contrastive_pairs(
        contrastive_instances,
        rng=rng,
        sample_at_most_n_pairs=max_contrastive_samples_per_concept_pair,
    )
    return train_instance_groups, contrastive_pairs


def split_instances(
    instances: Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
    *,
    rng: Random,
) -> Tuple[
    Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
    Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
]:
    """
    Randomly split all the given instances into two disjoint samples, returning the splits.
    """
    by_description = group_by_description(instances)

    split1: MutableSequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ] = []
    split2: MutableSequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ] = []
    for group in by_description.values():
        rng.shuffle(group)
        split1.extend(group[len(group) // 2 :])
        split2.extend(group[: len(group) // 2])

    return split1, split2


def make_contrastive_pairs(
    instances: Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
    *,
    rng: Random,
    sample_at_most_n_pairs: int,
) -> Sequence[
    Tuple[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ],
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ],
    ]
]:
    """
    Given a sequence of instances,

    We assume that

    - there are at least two different concepts represented among the instances
    - two instances teach the same concept if and only if they have the same description
        - subject to change when we implement this for actions :)
    """
    by_description = group_by_description(instances)

    result = []
    for description1, description2 in unique_pairs(by_description.keys()):
        description1_samples = list(by_description[description1])
        description2_samples = list(by_description[description2])
        all_paired_samples = list(product(description1_samples, description2_samples))
        rng.shuffle(all_paired_samples)
        result.extend(all_paired_samples[:sample_at_most_n_pairs])

    return result


def group_by_description(
    instances: Sequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
) -> Mapping[
    LinguisticDescriptionT,
    MutableSequence[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ],
]:
    """
    Group the given instances by their descriptions.
    """
    result = defaultdict(list)
    for instance in instances:
        _situation, description, _perception = instance
        result[description].append(instance)
    return dict(result)


def unique_pairs(it: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """
    Return a generator over the unique pairs in the given iterable.
    """
    seq = tuple(it)
    for idx, item in enumerate(seq):
        for other_item in seq[idx + 1 :]:
            yield item, other_item


if __name__ == "__main__":
    parameters_only_entry_point(contrastive_learning_entry_point)
