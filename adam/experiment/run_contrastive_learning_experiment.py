"""
Script to implement the contrastive learning experiment.

For now this only supports
"""
from collections import defaultdict
import logging
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
    NamedTuple,
    Generic,
)

from attr import evolve
from immutablecollections import immutableset
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum import ExplicitWithSituationInstanceGroup, InstanceGroup
from adam.experiment import execute_experiment
from adam.experiment.log_experiment import experiment_from_params
from adam.language import LinguisticDescriptionT
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
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.perception.perception_graph import GraphLogger, PerceptionGraph
from adam.situation import SituationT

SYMBOLIC = "symbolic"
SIMULATED = "simulated"
T = TypeVar("T")  # pylint:disable=invalid-name,bad-option-value


def contrastive_learning_entry_point(params: Parameters) -> None:
    if params.namespace("object_learner").string("learner_type") != "subset":
        raise NotImplementedError(
            "Contrastive learning only implemented for *subset* object learners."
        )
    if params.namespace_or_empty("attribute_learner"):
        raise NotImplementedError("Contrastive learning not implemented for attributes.")
    if params.namespace_or_empty("relation_learner"):
        raise NotImplementedError("Contrastive learning not implemented for relations.")
    if params.namespace_or_empty("action_learner"):
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

    base_experiment = experiment_from_params(params)
    logging.info("Setting up curriculum.")
    apprentice_train_instance_groups, contrastive_pairs = make_curriculum(
        [
            instance
            for group in base_experiment.training_stages
            for instance in group.instances()
        ],
        random_seed=params.integer("curriculum_creation_seed"),
    )

    experiment = evolve(base_experiment, training_stages=apprentice_train_instance_groups)
    logging.info("Training apprentice.")
    # We no longer support any non-integrated learners, but that hasn't made it into the type system
    # yet, so we need a cast here to tell MyPy we know what we're doing.
    learner = cast(
        IntegratedTemplateLearner[PerceptionT, LinguisticDescriptionT],
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
    contrastive_learner = TeachingContrastiveObjectLearner(apprentice=object_learner)
    for (_situation1, description1, perception1), (
        _situation2,
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
                learners=[object_learner],
            )
        )
    learner.log_hypotheses(
        params.creatable_directory("after_contrastive_hypothesis_log_dir")
    )
    logging.info("Contrastive learning finished.")


Instance = Tuple[
    Optional[SituationT],
    LinguisticDescriptionT,
    PerceptualRepresentation[PerceptionT],
]
ContrastiveInstance = Tuple[Instance, Instance]
LearningInstance = Tuple[LinguisticDescriptionT, PerceptionGraph]
ContrastiveLearningInstance = Tuple[LearningInstance, LearningInstance]


class AlignableExample(NamedTuple, Generic[LinguisticDescriptionT]):
    """
    An example that can be processed by some composable learners to produce a
    `LanguagePerceptionSemanticAlignment`.
    """

    linguistic_description: LinguisticDescriptionT
    perception_graph: PerceptionGraph


def make_contrast(
    pair: Tuple[AlignableExample, AlignableExample],
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
    instances: Sequence[Instance], *, random_seed: int
) -> Tuple[Sequence[InstanceGroup], Sequence[ContrastiveInstance]]:
    """
    Create a curriculum of (1) instances used to train the apprentice and (2) contrastive pairs for
    the contrastive learner.
    """
    rng = Random(random_seed)
    apprentice_train_instances, contrastive_instances = split_instances(
        instances, rng=rng
    )
    train_instance_groups = [
        ExplicitWithSituationInstanceGroup(
            name=description.as_token_string(), instances=instances  # type: ignore
        )
        for description, instances in group_by_description(
            apprentice_train_instances
        ).items()
    ]
    contrastive_pairs = make_contrastive_pairs(contrastive_instances, rng=rng)
    return train_instance_groups, contrastive_pairs


def split_instances(
    instances: Sequence[Instance], *, rng: Random
) -> Tuple[Sequence[Instance], Sequence[Instance]]:
    """
    Randomly split all the given instances into two disjoint samples, returning the splits.
    """
    by_description = group_by_description(instances)

    split1 = []
    split2 = []
    for group in by_description.values():
        rng.shuffle(group)
        split1.extend(group[len(group) // 2 :])
        split2.extend(group[: len(group) // 2])

    return split1, split2


def make_contrastive_pairs(
    instances: Sequence[Instance], *, rng: Random
) -> Sequence[ContrastiveInstance]:
    """
    Given a sequence of instances,

    We assume that

    - there are at least two different concepts represented among the instances
    - all concepts have the same number of samples
    - two instances teach the same concept if and only if they have the same description
        - subject to change when we implement this for actions :)
    """
    by_description = group_by_description(instances)
    check_descriptions_have_same_num_of_instances(
        by_description,  # type: ignore
        msg_when_check_fails="Expected all concepts to have the same number of samples.",
    )

    result = []
    for description1, description2 in unique_pairs(by_description.keys()):
        description1_samples = list(by_description[description1])
        description2_samples = list(by_description[description2])

        # Shuffle the samples so we can sample by popping from the end
        rng.shuffle(description1_samples)
        rng.shuffle(description2_samples)
        while description1_samples and description2_samples:
            sample1 = description1_samples.pop()
            sample2 = description2_samples.pop()
            result.append((sample1, sample2))

    return result


def check_descriptions_have_same_num_of_instances(
    by_description: Mapping[LinguisticDescriptionT, Instance],
    *,
    msg_when_check_fails: str,
) -> None:
    """
    Given a grouping of instances by description, make sure each group is the same size.

    This works by raising an with the given message when the check fails.
    """
    num_samples = None
    for description in by_description:
        if num_samples is None:
            num_samples = len(by_description[description])
        if len(by_description[description]) != num_samples:
            raise ValueError(msg_when_check_fails)


def group_by_description(
    instances: Sequence[Instance],
) -> Mapping[LinguisticDescriptionT, MutableSequence[Instance]]:
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
