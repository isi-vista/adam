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
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    sampled,
)
import random


def make_simple_pursuit_curriculum(
    *,
    num_instances: int = 10,
    num_noise_instances: int = 0,
    num_objects_in_instance: int = 3,
) -> Phase1InstanceGroup:
    """
    Creates a Pursuit-learning curriculum with for a set of standard objects. Each instance in the curriculum is a set
    of *num_objects_in_instance* objects paired with a word.
    We say an instance is non-noisy if the word refers to one of the objects in the set.
    An instance is noisy if none of the objects correspond to the word.
    For each type of object of interest, we will generate *num_instances_per_object_type* instances,
    of which *num_noise_instances_per_object_type* will be noisy.
    Args:
        num_instances: total number of learning instances for each object to learn
        num_noise_instances: number of noisy learning instances in each set of learning instances. A noisy instance in
        a scene where the utterance doesn't match the situation and perception (e.g. hearing "ball" while seeing a cup).
        num_objects_in_instance: number of objects in each instance
    """
    if num_noise_instances > num_instances:
        raise RuntimeError("Cannot have more noise than regular exemplars")

    target_objects = [BALL, CHAIR, PERSON, TABLE, DOG, BIRD, BOX]
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
            ).instances()
        )
        all_instances.extend(non_noise_instances)

        # Create instances for noise
        noise_instances = phase1_instances(
            "simple_pursuit_curriculum",
            sampled(
                noise_template,
                max_to_sample=num_noise_instances,
                chooser=PHASE1_CHOOSER,
                ontology=GAILA_PHASE_1_ONTOLOGY,
            ),
        ).instances()
        for (situation, _, perception) in noise_instances:
            # For each instance to be replaced by a noisy instance, we keep the correct utterance, but
            # replace the situation and perception. We do this to test the model's tolerance to varying degrees of
            # noise that reflects noisy instances in the real-world.
            linguistic_description = PHASE1_CHOOSER.choice(non_noise_instances)[1]
            all_instances.append((situation, linguistic_description, perception))

    description = f"simple_pursuit_curriculum_examples-{num_instances}_objects-{num_objects_in_instance}_noise-{num_noise_instances}"
    rng = random.Random()
    rng.seed(0)
    random.shuffle(all_instances, rng.random)
    final_instance_group: Phase1InstanceGroup = ExplicitWithSituationInstanceGroup(
        description, all_instances
    )
    return final_instance_group
