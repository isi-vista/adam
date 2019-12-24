"""
Curricula for Pursuit Learning Algorithm
In Pursuit, given a set of scenes and labels, the learner hypothesizes a meaning for the label and uses confidence
metrics to pursue the strongest hypothesis as long as it is supported by the following scenes.
Paper: The Pursuit of Word Meanings (Stevens et al., 2017)
"""
from adam.curriculum import ExplicitWithSituationInstanceGroup
from adam.curriculum.curriculum_utils import (
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
    standard_object,
)
from adam.ontology.phase1_ontology import (
    BIRD,
    BOX,
    GAILA_PHASE_1_ONTOLOGY,
    PERSON,
    BALL,
    CHAIR,
    TABLE,
    DOG,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    sampled,
)
import random


def make_simple_pursuit_curriculum(
    *,
    target_objects=[BALL, CHAIR, PERSON, TABLE, DOG, BIRD, BOX],
    num_instances: int = 10,
    num_noise_instances: int = 0,
    num_objects_in_instance: int = 3,
    perception_generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = GAILA_PHASE_1_PERCEPTION_GENERATOR,
) -> Phase1InstanceGroup:
    """
    Creates a Pursuit-learning curriculum with for a set of standard objects. Each instance in the curriculum is a set
    of *num_objects_in_instance* objects paired with a word.
    We say an instance is non-noisy if the word refers to one of the objects in the set.
    An instance is noisy if none of the objects correspond to the word.
    For each type of object of interest, we will generate *num_instances_per_object_type* instances,
    of which *num_noise_instances_per_object_type* will be noisy.
    """
    if num_noise_instances > num_instances:
        raise RuntimeError("Cannot have more noise than regular exemplars")

    noise_object_variables = [
        standard_object("obj-" + str(idx)) for idx in range(num_objects_in_instance)
    ]

    # A template that is used to replace situations and perceptions (not linguistic description) in noise instances
    noise_template = Phase1SituationTemplate(
        "simple_pursuit-noise",
        salient_object_variables=[noise_object_variables[0]],
        background_object_variables=noise_object_variables[1:],
    )

    all_instances = []
    # Generate phase_1 instance groups for each template (i.e each target word)
    for target_object in target_objects:
        target_object_variable = object_variable(
            target_object.handle + "-target", target_object
        )
        # For each target object, create a template with specific a target object in each to create learning instances.
        # There is one object (e.g. Ball) across all instances while the other objects vary. Hence, the target object is
        # a salient object (used for the linguistic description) while the remaining objects are background objects.
        object_is_present_template = Phase1SituationTemplate(
            "simple_pursuit",
            salient_object_variables=[target_object_variable],
            background_object_variables=noise_object_variables[:-1],
        )
        non_noise_instances = list(
            phase1_instances(
                "simple_pursuit_curriculum",
                sampled(
                    object_is_present_template,
                    max_to_sample=num_instances - num_noise_instances,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                perception_generator=perception_generator,
            ).instances()
        )

        # Filter out instances in which the target is present more than once, to ensure "a ball" instead of "the balls"
        for instance in non_noise_instances:
            # If the target appears exactly once (does not appear in background objects) keep using this instance
            situation = instance[0]
            if situation and not any(
                [obj.ontology_node == target_object for obj in situation.other_objects]
            ):
                all_instances.append(instance)

        # Create instances for noise
        noise_instances = phase1_instances(
            "simple_pursuit_curriculum",
            sampled(
                noise_template,
                max_to_sample=num_noise_instances,
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
            ),
            perception_generator=perception_generator,
        ).instances()
        # [1] is the index of the linguistic description in an instance
        # It doesn't matter which non-noise instance is chosen
        # because they all have the object type name as their linguistic description.
        target_object_linguistic_description = all_instances[-1][1]
        for (situation, _, perception) in noise_instances:
            # A noise instance needs to have the word for our target object
            # while not actually having our target object be present.
            # However, our language generator can't generate irrelevant language for a situation.
            # Therefore, we generate the instance as normal above,
            # but here we replace its linguistic description with the word for the target object.

            # Skip the noise instance if the target object appears in the noise data
            if situation and not any(
                [obj.ontology_node == target_object for obj in situation.all_objects]
            ):
                all_instances.append(
                    (situation, target_object_linguistic_description, perception)
                )

    description = (
        f"simple_pursuit_curriculum_examples-{num_instances}_objects-{num_objects_in_instance}_noise-"
        f"{num_noise_instances} "
    )
    rng = random.Random()
    rng.seed(0)
    random.shuffle(all_instances, rng.random)
    final_instance_group: Phase1InstanceGroup = ExplicitWithSituationInstanceGroup(
        description, all_instances
    )
    return final_instance_group


def make_pursuit_curriculum():
    return [
        make_simple_pursuit_curriculum(),
        make_simple_pursuit_curriculum(num_noise_instances=2),
        make_simple_pursuit_curriculum(num_objects_in_instance=4, num_noise_instances=2),
    ]
