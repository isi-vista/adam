import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import (
    _make_kind_predicates_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_generic_statements_curriculum,
    _make_eat_curriculum,
    _make_drink_curriculum,
    _make_sit_curriculum,
    _make_jump_curriculum,
    _make_fly_curriculum,
    _make_plural_objects_curriculum,
    _make_objects_with_colors_curriculum,
    _make_colour_predicates_curriculum,
)
from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_utils import phase2_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    semantics_as_weighted_adjacency_matrix,
    evaluate_kind_membership,
    cos_sim,
)
from adam.learner.objects import PursuitObjectLearnerNew, SubsetObjectLearnerNew
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, GROUND
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.semantics import Concept, ObjectConcept
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        # object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        object_learner=SubsetObjectLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        attribute_learner=SubsetAttributeLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        plural_learner=SubsetPluralLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        action_learner=SubsetVerbLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        generics_learner=SimpleGenericsLearner(),
    )


def run_experiment(learner, curricula, experiment_id):
    # Teach each pretraining curriculum
    for curriculum in curricula:
        print('Teaching', curriculum.name())
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - prepositison learner can't learn without already recognized objects
            # print(' '.join(linguistic_description.as_token_sequence()))
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    # learner.object_learner.log_hypotheses(Path(f"./{experiment_id}-{type(learner.object_learner)}"))
    # learner.log_hypotheses(Path(f"./{experiment_id}-{type(learner.object_learner)}"))


    # Teach each kind member
    empty_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_2_ONTOLOGY,
        salient_objects=[
            SituationObject.instantiate_ontology_node(
                ontology_node=GROUND,
                debug_handle=GROUND.handle,
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
        ],
    )
    empty_perception = GAILA_PHASE_2_PERCEPTION_GENERATOR.generate_perception(
        empty_situation, PHASE1_CHOOSER_FACTORY()
    )
    pseudoword_to_kind = {"wug": "animal", "vonk": "food", "snarp": "people"}
    for word, kind in pseudoword_to_kind.items():
        print(word, "s", "are", kind, "s")
        learner.observe(
            LearningExample(
                empty_perception,
                TokenSequenceLinguisticDescription(tokens=(word, "s", "are", kind, "s"))
                if kind != "people"
                else TokenSequenceLinguisticDescription(
                    tokens=(word, "s", "are", kind, "s")
                ),
            )
        )

    complete_results = []
    print('Results for ', experiment_id)
    for word, gold_kind in pseudoword_to_kind.items():
        results = [
            (kind, evaluate_kind_membership(learner.semantics_graph, word, kind))
            for kind in pseudoword_to_kind.values()
        ]
        complete_results.append(results)

    results_df = pd.DataFrame([[np.asscalar(i[1]) for i in l] for l in complete_results], columns=['Animal', 'Food', 'People'])
    results_df.insert(0,'Words',pseudoword_to_kind.keys())
    print(results_df.to_csv(index=False))
    learner.log_hypotheses(Path(f"./renders/{experiment_id}"))

    # embeddings = semantics_as_weighted_adjacency_matrix(learner.semantics_graph)
    # objects_to_embeddings = {
    #     n: embeddings[i]
    #     for i, n in enumerate(learner.semantics_graph.nodes)
    #     if isinstance(n, ObjectConcept)
    # }
    # generate_heatmap(objects_to_embeddings, experiment_id)
    #
    # generate_similarities(semantic_matrix, list(learner.semantics_graph.nodes()), ObjectConcept)

    # learner.render_semantics_to_file(
    #     graph=learner.semantics_graph,
    #     graph_name="semantics",
    #     output_file=Path(f"./renders/{experiment_id}/semantics.png"),
    # )


def generate_heatmap(nodes_to_embeddings: Dict[Concept, Any], filename: str):
    if not nodes_to_embeddings:
        return
    similarity_matrix = np.zeros((len(nodes_to_embeddings), len(nodes_to_embeddings)))
    for i, (_, embedding_1) in enumerate(nodes_to_embeddings.items()):
        for j, (_, embedding_2) in enumerate(nodes_to_embeddings.items()):
            similarity_matrix[i][j] = cos_sim(embedding_1, embedding_2)
    names = [n.debug_string for n in nodes_to_embeddings.keys()]
    df = pd.DataFrame(data=similarity_matrix, index=names, columns=names)
    plt.rcParams["figure.figsize"] = (20.0, 20.0)
    plt.rcParams["font.family"] = "serif"
    # sb.heatmap(df)
    # sb.clustermap(df)
    sb.clustermap(df, row_cluster=True, col_cluster=True)
    # cm.ax_row_dendrogram.set_visible(False)
    # cm.ax_col_dendrogram.set_visible(False)
    # plt.show()
    plt.savefig(f"plots/{filename}.png")
    plt.close()


if __name__ == "__main__":
    for lm in [LanguageMode.ENGLISH]:
        language_generator = phase2_language_generator(lm)
        for num_samples in [10, 25, 50]:
            pretraining_curriculas = {
                # "just-objects": [
                #     _make_each_object_by_itself_curriculum(
                #         num_samples, 0, language_generator
                #     )
                # ],
                # "objects-and-kinds": [
                #     _make_each_object_by_itself_curriculum(
                #         num_samples, 0, language_generator
                #     ),
                #     _make_kind_predicates_curriculum(None, None, language_generator),
                # ],
                "kinds-and-generics": [
                    _make_each_object_by_itself_curriculum(
                        num_samples, 0, language_generator
                    ),
                    # _make_plural_objects_curriculum(num_samples, 0, language_generator),
                    _make_kind_predicates_curriculum(None, None, language_generator),
                    _make_generic_statements_curriculum(
                        num_samples=3,
                        noise_objects=0,
                        language_generator=language_generator,
                    ),
                ],
                "obj-actions-kinds-generics": [
                    _make_each_object_by_itself_curriculum(
                        num_samples, 0, language_generator
                    ),
                    # Actions - verbs in generics
                    _make_eat_curriculum(10, 0, language_generator),
                    _make_drink_curriculum(10, 0, language_generator),
                    _make_sit_curriculum(10, 0, language_generator),
                    _make_jump_curriculum(10, 0, language_generator),
                    _make_fly_curriculum(10, 0, language_generator),
                    # Plurals
                    _make_plural_objects_curriculum(None, 0, language_generator),
                    # Color attributes
                    _make_objects_with_colors_curriculum(None, None, language_generator),
                    # Predicates
                    _make_colour_predicates_curriculum(None, None, language_generator),
                    _make_kind_predicates_curriculum(None, None, language_generator),
                    # Generics
                    _make_generic_statements_curriculum(
                        num_samples=3,
                        noise_objects=0,
                        language_generator=language_generator,
                    ),
                ],
                # build_gaila_m13_curriculum(num_samples=num_samples, num_noise_objects=0, language_generator=language_generator)
            }

            for curricula_name, pretraining_curricula in pretraining_curriculas.items():
                # Run experiment
                experiment = f"kind_semantics_lang-{lm}_num-samples-{num_samples}_cur-{curricula_name}"
                print("\nRunning experiment:", experiment)
                integrated_learner = integrated_learner_factory(lm)
                run_experiment(
                    learner=integrated_learner,
                    curricula=pretraining_curricula,
                    experiment_id=experiment,
                )
