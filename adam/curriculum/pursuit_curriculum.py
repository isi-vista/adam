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

random.seed(0)


def make_simple_pursuit_curriculum(
    *,
    num_instances: int = 10,
    num_noise_instances: int = 0,
    num_objects_in_instance: int = 3,
) -> Phase1InstanceGroup:
    """
    Creates a Pursuit-learning curriculum with for a set of standard objects.
    Args:
        num_instances: total number of learning instances for each object to learn
        num_noise_instances: number of noisy learning instances in each set of learning instances. A noisy instance in
        a scene where the utterance doesn't match the situation and perception (e.g. hearing "ball" while seeing a cup).
        num_objects_in_instance: number of objects in each instance
    """
    if num_noise_instances > num_instances:
        raise RuntimeError("Cannot have more noise than regular exemplars")

    target_objects = [BALL, CHAIR, PERSON, TABLE, DOG, BIRD, BOX]
    target_object_variables = [
        object_variable(target.handle + "-target", target) for target in target_objects
    ]
    other_objects = [
        standard_object("obj-" + str(idx)) for idx in range(num_objects_in_instance)
    ]

    # A list of templates with specific a target object in each to create learning instances. Specifically,
    # there is one object (e.g. Ball) across all instances while the other objects vary. Hence, the target object is
    # a salient object (used for generating a dependency tree) while the remaining objects are background objects.
    object_is_present_templates = [
        Phase1SituationTemplate(
            "simple_pursuit",
            salient_object_variables=[target_object_variable],
            background_object_variables=other_objects[:-1],
        )
        for target_object_variable in target_object_variables
    ]

    # A generic template that is used to replace situations and perceptions (not dependency trees) in noise instances
    noise_template = Phase1SituationTemplate(
        "simple_pursuit",
        salient_object_variables=[other_objects[0]],
        background_object_variables=other_objects[1:],
    )

    all_instances = []
    # Generate phase_1 instance groups for each template (i.e each target word)
    for object_is_present_template in object_is_present_templates:
        new_instances = list(
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
        all_instances.extend(new_instances)
        # Create instances for noise
        if num_noise_instances > 0:
            noise_instances = phase1_instances(
                "simple_pursuit_curriculum",
                sampled(
                    noise_template,
                    max_to_sample=num_noise_instances,
                    chooser=PHASE1_CHOOSER,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
            ).instances()
            for noise_instance in noise_instances:
                # For each instance to be replaced by a noisy instance, we keep the correct utterance, but
                # replace the situation and perception.
                dependency_tree = PHASE1_CHOOSER.choice(new_instances)[1]
                all_instances.append(
                    (noise_instance[0], dependency_tree, noise_instance[2])
                )

    description = f"simple_pursuit_curriculum_examples-{num_instances}_objects-{num_objects_in_instance}_noise-{num_noise_instances}"
    random.shuffle(all_instances)
    final_instance_group: Phase1InstanceGroup = ExplicitWithSituationInstanceGroup(
        description, all_instances
    )
    return final_instance_group
