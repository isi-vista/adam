from functools import partial
import logging
import shutil

import yaml

from PIL import Image, ImageFont, ImageDraw
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, MutableMapping, Optional, Tuple

from attr import attrib, attrs
from attr.validators import instance_of, is_callable, optional
from more_itertools import only, take
from adam.perception.visual_perception import VisualPerceptionRepresentation
from vistautils.parameters import Parameters

from adam.curriculum_to_html import CurriculumToHtmlDumper
from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.language.dependency import LinearizedDependencyTree
from adam.learner import TopLevelLanguageLearnerDescribeReturn
from adam.paths import SITUATION_DIR_NAME, ROBOTO_FILE
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.semantics import SemanticNode, ActionSemanticNode, RelationSemanticNode
from adam.situation import SituationT
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.phase_3_situations import SimulationSituation

GOLD_LANGUAGE = "gold_language"
OUTPUT_LANGUAGE = "output_language"


class DescriptionObserver(Generic[SituationT, LinguisticDescriptionT, PerceptionT], ABC):
    r"""
    Something which can observe the descriptions produced by `TopLevelLanguageLearner`\ s.

    Typically a `DescriptionObserver` will provide some sort of summary of its observations
    when its *report* method is called.
    """

    @abstractmethod
    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        r"""
        Observe a description provided by a `TopLevelLanguageLearner`.

        Args:
            situation: The `Situation` being described. This is optional.
            true_description: The "gold-standard" description of the situation.
            perceptual_representation: The `PerceptualRepresentation` of the situation received by
                                       the `TopLevelLanguageLearner`.
            predicted_scene_description:  `TopLevelLanguageLearnerDescribeReturn`
        """

    @abstractmethod
    def report(self) -> None:
        """
        Take some action based on the observations.

        Typically, this will be to write a report either to the console, to a file, or both.
        """


@attrs(slots=True)
class TopChoiceExactMatchObserver(
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Log how often the top-scoring predicted `LinguisticDescription` for a `Situation` exactly
    matches the expected `LinguisticDescription`.

    If there are multiple predicted `LinguisticDescription`\ s with the same score, which one is
    compared to determine matching is undefined.
    """
    name: str = attrib(validator=instance_of(str))
    _num_observations: int = attrib(init=False, default=0)
    _num_top_choice_matches: int = attrib(init=False, default=0)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        self._num_observations += 1

        if predicted_scene_description:
            top_choice = max(
                predicted_scene_description.description_to_confidence.items(),
                key=_by_score,
            )

            if top_choice == true_description:
                self._num_top_choice_matches += 1

    def report(self) -> None:
        logging.info(
            f"{self.name}: top prediction matched expected prediction "
            f"{self._num_observations} / {self._num_top_choice_matches} times ("
            f"{100.0*self._num_top_choice_matches/self._num_observations:2.3f} %)"
        )


@attrs(slots=True)
class CandidateAccuracyObserver(
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Log how often the 'gold' description is present in the learner's candidate descriptions.
    Provide an accuracy score.
    """
    name: str = attrib(validator=instance_of(str))

    # these params allow the accuracy to be written out to a text file at each step which is helpful for graphing for
    # experiments such as the gaze ablation where we want to compare accuracy
    accuracy_to_txt: bool = attrib(default=False)
    txt_path: str = attrib(validator=instance_of(str), default="accuracy_out.txt")
    _num_predictions: int = attrib(init=False, default=0)
    _num_predictions_with_gold_in_candidates: int = attrib(init=False, default=0)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        self._num_predictions += 1

        descriptions_as_token_sequences = [
            desc.as_token_sequence()
            for desc in predicted_scene_description.description_to_confidence
        ]
        if true_description.as_token_sequence() in descriptions_as_token_sequences:
            self._num_predictions_with_gold_in_candidates += 1

    def report(self) -> None:
        accuracy = self.accuracy()

        # write out to an accuracy file if requested to do so by the user
        if self.accuracy_to_txt:
            try:
                with open(self.txt_path, "a", encoding="utf-8") as f:
                    f.write(f"{accuracy}\n")
            # we currently catch errors with a warning rather than stopping the program if we can't log accuracy
            except OSError as e:
                logging.warning(
                    f"The following error occurred while attempting to log accuracy to a txt file: {e}"
                )

        if accuracy is not None:
            logging.info(
                "%s: accuracy of learner's predictions ('gold' description was in learner's candidates) "
                "%d / %d predictions ("
                "%03.2f %%)",
                self.name,
                self._num_predictions_with_gold_in_candidates,
                self._num_predictions,
                100 * accuracy,
            )

    def accuracy(self) -> Optional[float]:
        """
        Return accuracy value of the number of predictions made where the 'gold' description was present in
        the learner's candidate descriptions.
        Returns: accuracy score (float)

        """
        if not self._num_predictions:
            return None
        return self._num_predictions_with_gold_in_candidates / self._num_predictions


@attrs(slots=True)
class PrecisionRecallObserver(
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Log information to calculate the learners precision and recall
    """
    name: str = attrib(validator=instance_of(str))

    _num_predictions: int = attrib(init=False, default=0)
    _num_positive_examples: int = attrib(init=False, default=0)
    _num_true_positive_examples: int = attrib(init=False, default=0)
    _num_false_negative_examples: int = attrib(init=False, default=0)

    # these params allow the precision and recoll to be written out to a text file at each step
    # which is helpful for graphing for experiments
    make_report: bool = attrib(default=False, kw_only=True)
    txt_path: str = attrib(
        validator=instance_of(str), default="accuracy_out.txt", kw_only=True
    )
    robust: bool = attrib(default=True, kw_only=True)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        self._num_predictions += 1

        descriptions_as_token_sequences = [
            desc.as_token_sequence()
            for desc in predicted_scene_description.description_to_confidence
        ]

        if (
            isinstance(true_description, LinearizedDependencyTree)
            and not true_description.accurate
        ):
            # This means we have a false description, we blindly assume any `LinguisticDescription` that isn't of this type
            # Must be an accurate description of the situation - This is an eval hack as to avoid reworking the rest of the
            # Curriculum generation. See:
            if true_description.as_token_sequence() in descriptions_as_token_sequences:
                self._num_false_negative_examples += 1
        else:
            # This means the linguistic description is true for the situation
            self._num_positive_examples += 1
            if true_description.as_token_sequence() in descriptions_as_token_sequences:
                self._num_true_positive_examples += 1

    def report(self) -> None:
        precision = self.precision()
        recall = self.recall()

        # write out to an accuracy file if requested to do so by the user
        if self.make_report:
            try:
                with open(self.txt_path, "a", encoding="utf-8") as f:
                    f.write(f"{precision},{recall}\n")
            # we currently catch errors with a warning rather than stopping the program if we can't log accuracy
            except OSError as e:
                logging.warning(
                    f"The following error occurred while attempting to log accuracy to a txt file: {e}"
                )

        if precision is not None and recall is not None:
            logging.info(
                "%s: Precision of learner's predictions is %d / %d predictions (%03.2f %%)\n"
                "Recall of learner's predictions is %d / %d predictions (%03.2f %%)",
                self.name,
                self._num_true_positive_examples,
                self._num_positive_examples,
                100 * precision,
                self._num_true_positive_examples,
                self._num_true_positive_examples + self._num_false_negative_examples,
                100 * recall,
            )

    def precision(self) -> Optional[float]:
        if not self._num_positive_examples:
            return None
        else:
            return self._num_true_positive_examples / self._num_positive_examples

    def recall(self) -> Optional[float]:
        num_examples = (
            self._num_true_positive_examples + self._num_false_negative_examples
        )
        if not num_examples:
            return None
        else:
            return self._num_true_positive_examples / (
                self._num_true_positive_examples + self._num_false_negative_examples
            )


CSS = """
body {
    font-size: 1em;
    font-family: sans-serif;
}

table td { p
    padding: 1em; 
    background-color: #FAE5D3 ;
}
"""


@attrs(slots=True)
class LearningProgressHtmlLogger:  # pragma: no cover

    output_file_str: str = attrib(validator=instance_of(str))
    html_dumper: CurriculumToHtmlDumper = attrib(
        validator=instance_of(CurriculumToHtmlDumper)
    )
    include_links_to_images: bool = attrib(
        validator=instance_of(bool), kw_only=True, default=False
    )
    pre_observed_description: Optional[str] = attrib(init=False, default=None)

    _button_id_suffix: int = attrib(init=False, default=0)
    _num_pretty_descriptions: int = attrib(kw_only=True, default=3)
    _sort_by_length: bool = attrib(kw_only=True, default=False)

    @staticmethod
    def create_logger(params: Parameters) -> "LearningProgressHtmlLogger":
        output_dir = params.creatable_directory("experiment_group_dir")
        experiment_name = params.string("experiment")
        include_links_to_images = params.optional_boolean("include_image_links")
        num_pretty_descriptions = params.positive_integer(
            "num_pretty_descriptions", default=3
        )
        sort_by_length = params.boolean(
            "sort_learner_descriptions_by_length", default=False
        )

        logging_dir = output_dir / experiment_name
        logging_dir.mkdir(parents=True, exist_ok=True)
        output_html_path = logging_dir / "index.html"

        if include_links_to_images is None:
            include_links_to_images = False

        logging.info("Experiment will be logged to %s", output_html_path)

        html_dumper = CurriculumToHtmlDumper()
        if not output_html_path.exists():
            with output_html_path.open("w") as outfile:

                outfile.write(f"<head>\n\t<style>{CSS}\n\t</style>\n</head>")
                outfile.write(f"\n<body>\n\t<h1>{experiment_name}</h1>")
                # A JavaScript function to allow toggling perception information
                outfile.write(
                    """
                    <script>
                    function myFunction(id) {
                      var x = document.getElementById(id);
                      if (x.style.display === "none") {
                        x.style.display = "block";
                      } else {
                        x.style.display = "none";
                      }
                    }
                    </script>
                    """
                )
        return LearningProgressHtmlLogger(
            output_file_str=str(output_html_path),
            html_dumper=html_dumper,
            include_links_to_images=include_links_to_images,
            num_pretty_descriptions=num_pretty_descriptions,
            sort_by_length=sort_by_length,
        )

    def pre_observer(
        self,
        *,
        params: Parameters = Parameters.empty(),
        experiment_group_dir: Optional[Path] = None,
    ) -> "DescriptionObserver":  # type: ignore
        track_accuracy = params.boolean("include_acc_observer", default=False)
        log_accuracy = params.boolean("accuracy_to_txt", default=False)
        log_accuracy_path = params.string(
            "accuracy_logging_path",
            default=f"{experiment_group_dir}/accuracy_pre_out.txt"
            if experiment_group_dir
            else "accuracy_pre_out.txt",
        )
        track_precision_recall = params.boolean("include_pr_observer", default=False)
        log_precision_recall = params.boolean("log_pr", default=False)
        log_precision_recall_path = params.string(
            "pr_log_path",
            default=f"{experiment_group_dir}/pr_post_out.txt"
            if experiment_group_dir
            else "pr_post_out.txt",
        )
        return HTMLLoggerPreObserver(
            name="Pre-observer",
            html_logger=self,
            candidate_accuracy_observer=CandidateAccuracyObserver(
                name="Pre-observer-acc",
                accuracy_to_txt=log_accuracy,
                txt_path=log_accuracy_path,
            )
            if track_accuracy
            else None,
            precision_recall_observer=PrecisionRecallObserver(
                name="Pre-observer-pr",
                make_report=log_precision_recall,
                txt_path=log_precision_recall_path,
            )
            if track_precision_recall
            else None,
        )

    def post_observer(
        self,
        *,
        params: Parameters = Parameters.empty(),
        experiment_group_dir: Optional[Path] = None,
    ) -> "DescriptionObserver":  # type: ignore
        # these are the params to use for writing accuracy to a text file at every iteration (e.g. to graph later)
        track_accuracy = params.boolean("include_acc_observer", default=True)
        log_accuracy = params.boolean("accuracy_to_txt", default=False)
        log_accuracy_path = params.string(
            "accuracy_logging_path",
            default=f"{experiment_group_dir}/accuracy_post_out.txt"
            if experiment_group_dir
            else "accuracy_post_out.txt",
        )
        track_precision_recall = params.boolean("include_pr_observer", default=False)
        log_precision_recall = params.boolean("log_pr", default=False)
        log_precision_recall_path = params.string(
            "pr_log_path",
            default=f"{experiment_group_dir}/pr_post_out.txt"
            if experiment_group_dir
            else "pr_post_out.txt",
        )
        return HTMLLoggerPostObserver(
            name="Post-observer",
            html_logger=self,
            candidate_accuracy_observer=CandidateAccuracyObserver(
                name="Post-observer-acc",
                accuracy_to_txt=log_accuracy,
                txt_path=log_accuracy_path,
            )
            if track_accuracy
            else None,
            precision_recall_observer=PrecisionRecallObserver(
                name="Post-observer-pr",
                make_report=log_precision_recall,
                txt_path=log_precision_recall_path,
            )
            if track_precision_recall
            else None,
            test_mode=False,
        )

    def test_observer(
        self,
        *,
        params: Parameters = Parameters.empty(),
        experiment_group_dir: Optional[Path] = None,
    ) -> "DescriptionObserver":  # type: ignore
        # these are the params to use for writing accuracy to a text file at every iteration (e.g. to graph later)
        track_accuracy = params.boolean("include_acc_observer", default=True)
        log_accuracy = params.boolean("accuracy_to_txt", default=False)
        log_accuracy_path = params.string(
            "accuracy_logging_path",
            default=f"{experiment_group_dir}/accuracy_test_out.txt"
            if experiment_group_dir
            else "accuracy_test_out.txt",
        )
        track_precision_recall = params.boolean("include_pr_observer", default=False)
        log_precision_recall = params.boolean("log_pr", default=False)
        log_precision_recall_path = params.string(
            "pr_log_path",
            default=f"{experiment_group_dir}/pr_test_out.txt"
            if experiment_group_dir
            else "pr_test_out.txt",
        )
        accuracy_observer = None
        precision_recall_observer = None
        if track_accuracy:
            accuracy_observer = CandidateAccuracyObserver(
                name="Test-observer-acc",
                accuracy_to_txt=log_accuracy,
                txt_path=log_accuracy_path,
            )
        if track_precision_recall:
            precision_recall_observer = PrecisionRecallObserver(
                name="Test-observer-pr",
                make_report=log_precision_recall,
                txt_path=log_precision_recall_path,
            )

        return HTMLLoggerPostObserver(
            name="t-observer",
            html_logger=self,
            candidate_accuracy_observer=accuracy_observer,
            precision_recall_observer=precision_recall_observer,
            test_mode=True,
        )

    def pre_observer_log(
        self,
        predicted_descriptions: TopLevelLanguageLearnerDescribeReturn,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
    ) -> None:
        append_str = ""
        if accuracy:
            append_str += f"\nAccuracy: {accuracy:2.2f}"
        if precision:
            append_str += f"\nPrecision: {precision:2.2f}"
        if recall:
            append_str += f"\nRecall: {recall:2.2f}"
        self.pre_observed_description = (
            pretty_descriptions(
                predicted_descriptions.description_to_confidence,
                self._num_pretty_descriptions,
                sort_by_length=self._sort_by_length,
            )
            + append_str
        )

    def post_observer_log(
        self,
        *,
        observer_name: str,
        instance_number: int,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: TopLevelLanguageLearnerDescribeReturn,
        test_mode: bool,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
    ):
        learner_pre_description = self.pre_observed_description
        self.pre_observed_description = None

        learner_description = pretty_descriptions(
            predicted_descriptions.description_to_confidence,
            self._num_pretty_descriptions,
            sort_by_length=self._sort_by_length,
        )

        if situation and isinstance(situation, HighLevelSemanticsSituation):
            situation_text, _ = self.html_dumper.situation_text(situation)
        else:
            situation_text = ""

        perception_text = self.html_dumper.perception_text(  # type: ignore
            perceptual_representation  # type: ignore
        )

        true_description_text = " ".join(true_description.as_token_sequence())

        clickable_perception_string = f"""
            <button onclick="myFunction('myPerception{instance_number}')">View Perception</button>
            <div id="myPerception{instance_number}" style="display: none">
            {perception_text}
            </div>
            """

        # Log into html file
        # We want to log the true description, the learners guess, the perception, and situation
        with open(self.output_file_str, "a+", encoding="utf-8") as outfile:
            outfile.write(
                f"\n\t<table>\n"
                f"\t\t<thead>\n"
                f"\t\t\t<tr>\n"
                f'\t\t\t\t<th colspan="3">\n'
                f"\t\t\t\t\t<h2>Learning Instance: {observer_name} report number {instance_number}</h2>\n"
                f"\t\t\t\t</th>\n\t\t\t</tr>\n"
                f"\t\t</thead>\n"
                f"\t\t<tbody>\n"
                f"\t\t\t<tr>\n"
                f"\t\t\t\t<td>\n"
                f'\t\t\t\t\t<h3 id="situation-{instance_number}">Situation</h3>\n'
                f"\t\t\t\t</td>\n"
                f"\t\t\t\t<td>\n"
                f'\t\t\t\t\t<h3 id="true-{instance_number}">True Description</h3>\n'
                f"\t\t\t\t</td>\n"
                f"\t\t\t\t<td>\n"
            )
            if test_mode:
                # in test mode we don't update the learner, so there is no pre- and
                # post-description, just a single description.
                outfile.write(
                    f'\t\t\t\t\t<h3 id="learner-pre-{instance_number}">Learner\'s Description</h3>\n'
                    f"\t\t\t\t</td>\n"
                )
            else:
                outfile.write(
                    f'\t\t\t\t\t<h3 id="learner-pre-{instance_number}">Learner\'s Old Description</h3>\n'
                    f"\t\t\t\t</td>\n"
                    f"\t\t\t\t<td>\n"
                    f'\t\t\t\t\t<h3 id="learner-post-{instance_number}">Learner\'s New Description</h3>\n'
                    f"\t\t\t\t</td>\n"
                )
            outfile.write(
                f"\t\t\t\t<td>\n"
                f'\t\t\t\t\t<h3 id="perception-{instance_number}">Learner Perception</h3>\n'
                f"\t\t\t\t</td>\n"
            )
            if self.include_links_to_images:
                outfile.write(
                    "\t\t\t\t\t<td>\n"
                    "\t\t\t\t\t<h3>Scene Renderings</h3>\n"
                    "\t\t\t\t\t</td>\n"
                )
            outfile.write(
                f"\t\t\t</tr>\n"
                f"\t\t\t<tr>\n"
                f'\t\t\t\t<td valign="top">{situation_text}\n\t\t\t\t</td>\n'
                f'\t\t\t\t<td valign="top">{true_description_text}</td>\n'
            )
            composit_learner_description = (
                f'\t\t\t\t<td valign="top">{learner_description}'
            )
            if accuracy:
                composit_learner_description = (
                    composit_learner_description + f"<br/>Accuracy: {accuracy:2.2f}"
                )
            if precision:
                composit_learner_description = (
                    composit_learner_description + f"<br/>Precision: {precision:2.2f}"
                )
            if recall:
                composit_learner_description = (
                    composit_learner_description + f"<br/>Recall: {recall:2.2f}"
                )
            composit_learner_description = composit_learner_description + "</td>\n"
            if test_mode:
                outfile.write(f"{composit_learner_description}")
            else:
                outfile.write(
                    f'\t\t\t\t<td valign="top">{learner_pre_description}</td>\n'
                    f"{composit_learner_description}"
                )

            render_buttons_text = ""

            outfile.write(
                f'\t\t\t\t<td valign="top">{clickable_perception_string}\n\t\t\t\t</td>\n'
            )
            if self.include_links_to_images:
                outfile.write(f"\t\t\t\t<td valign='top'>{render_buttons_text}</td>")
            outfile.write("\t\t\t</tr>\n\t\t</tbody>\n\t</table>")
            outfile.write("\n</body>")

    def _get_button_suffix(self) -> str:
        suffix = str(self._button_id_suffix)
        self._button_id_suffix += 1
        return suffix


@attrs(slots=True)
class HTMLLoggerPreObserver(  # pragma: no cover
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Logs the true description and learner's descriptions throughout the learning process.
    """
    name: str = attrib(validator=instance_of(str))
    html_logger: LearningProgressHtmlLogger = attrib(
        init=True, validator=instance_of(LearningProgressHtmlLogger), kw_only=True
    )

    # fmt: off
    candidate_accuracy_observer: Optional[CandidateAccuracyObserver] = attrib(  # type: ignore
        kw_only=True,
        validator=optional(instance_of(CandidateAccuracyObserver)),
    )

    precision_recall_observer: Optional[PrecisionRecallObserver] = attrib(  # type: ignore
        kw_only=True,
        validator=optional(instance_of(PrecisionRecallObserver)),
    )
    # fmt: on

    def observe(  # pylint: disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        if self.candidate_accuracy_observer:
            self.candidate_accuracy_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        if self.precision_recall_observer:
            self.precision_recall_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        self.html_logger.pre_observer_log(
            predicted_scene_description,
            accuracy=self.candidate_accuracy_observer.accuracy()
            if self.candidate_accuracy_observer
            else None,
            precision=self.precision_recall_observer.precision()
            if self.precision_recall_observer
            else None,
            recall=self.precision_recall_observer.recall()
            if self.precision_recall_observer
            else None,
        )

    def report(self) -> None:
        if self.candidate_accuracy_observer:
            self.candidate_accuracy_observer.report()
        if self.precision_recall_observer:
            self.precision_recall_observer.report()


@attrs(slots=True)
class HTMLLoggerPostObserver(  # pragma: no cover
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Logs the true description and learner's descriptions throughout the learning process.
    """
    name: str = attrib(validator=instance_of(str))
    html_logger: LearningProgressHtmlLogger = attrib(
        validator=instance_of(LearningProgressHtmlLogger), kw_only=True
    )
    candidate_accuracy_observer = attrib(kw_only=True)  # type: ignore
    precision_recall_observer = attrib(kw_only=True)  # type: ignore
    test_mode: bool = attrib(validator=instance_of(bool), kw_only=True)
    counter: int = attrib(kw_only=True, default=0)

    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        if self.candidate_accuracy_observer:
            self.candidate_accuracy_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        if self.precision_recall_observer:
            self.precision_recall_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        self.html_logger.post_observer_log(
            observer_name=self.name,
            instance_number=self.counter,
            situation=situation,
            true_description=true_description,
            perceptual_representation=perceptual_representation,
            predicted_descriptions=predicted_scene_description,
            test_mode=self.test_mode,
            accuracy=self.candidate_accuracy_observer.accuracy()
            if self.candidate_accuracy_observer
            else None,
            precision=self.precision_recall_observer.precision()
            if self.precision_recall_observer
            else None,
            recall=self.precision_recall_observer.recall()
            if self.precision_recall_observer
            else None,
        )
        self.counter += 1

    def report(self) -> None:
        if self.candidate_accuracy_observer:
            self.candidate_accuracy_observer.report()
        if self.precision_recall_observer:
            self.precision_recall_observer.report()


@attrs(slots=True)
class ByLanguageCandidateAccuracyObserver(
    DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]
):
    """
    Like CandidateAccuracyObserver except we calculate accuracy separately for each true
    description.

    This is used for example to calculate objectwise accuracy values for M5.
    """

    name: str = attrib(validator=instance_of(str))
    candidate_accuracy_observer_factory: Callable[
        [str], CandidateAccuracyObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(kw_only=True, validator=is_callable())
    true_description_to_observer: MutableMapping[
        str, CandidateAccuracyObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(init=False, factory=dict)

    @staticmethod
    def _make_candidate_accuracy_observer(
        own_name: str,
        true_language: str,
    ):
        return CandidateAccuracyObserver(f"{own_name}_for_{true_language}")

    @candidate_accuracy_observer_factory.default
    def _default_candidate_accuracy_observer_factory(
        self,
    ) -> Callable[
        [str], CandidateAccuracyObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ]:
        return partial(
            ByLanguageCandidateAccuracyObserver._make_candidate_accuracy_observer,
            self.name,
        )

    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        true_description_string = " ".join(true_description.as_token_sequence())
        observer_for_description = self.true_description_to_observer.setdefault(
            true_description_string,
            self.candidate_accuracy_observer_factory(true_description_string),
        )
        observer_for_description.observe(
            situation,
            true_description,
            perceptual_representation,
            predicted_scene_description,
        )

    def report(self) -> None:
        for true_description, observer in self.true_description_to_observer.items():
            # Why report accuracy here instead of using the observer's .report()? Because we want to
            # specify in *this message* "This is the accuracy for scenes described as 'a ball'"
            # in one message rather than having "true description: a ball" followed by an accuracy
            # report (which could get hard to read).
            accuracy = observer.accuracy()

            if accuracy is not None:
                # pylint:disable=protected-access
                logging.info(
                    "%s: for description '%s', accuracy of learner's predictions ('gold' "
                    "description was in learner's candidates) %d / %d predictions ("
                    "%03.2f %%)",
                    self.name,
                    true_description,
                    # hack :(
                    observer._num_predictions_with_gold_in_candidates,  # noqa
                    observer._num_predictions,  # noqa
                    100 * accuracy,
                )


@attrs(slots=True)
class YAMLLogger(DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]):
    name: str = attrib(validator=instance_of(str))
    experiment_path: Path = attrib(validator=instance_of(Path))
    counter: int = attrib(kw_only=True, default=0)
    copy_curriculum: bool = attrib(kw_only=True, default=True)
    file_name: Optional[str] = attrib(validator=optional(instance_of(str)), default=None)
    _by_language_candidate_accuracy_observer: Optional[
        ByLanguageCandidateAccuracyObserver[
            SituationT, LinguisticDescriptionT, PerceptionT
        ]
    ] = attrib(
        validator=optional(instance_of(ByLanguageCandidateAccuracyObserver)), default=None
    )
    _candidate_accuracy_observer: Optional[
        CandidateAccuracyObserver[SituationT, LinguisticDescriptionT, PerceptionT]
    ] = attrib(validator=optional(instance_of(CandidateAccuracyObserver)), default=None)

    @staticmethod
    def from_params(
        name: str, params: Parameters
    ) -> Optional["YAMLLogger[SituationT, LinguisticDescriptionT, PerceptionT]"]:
        if "experiment_output_path" not in params:
            return None
        return YAMLLogger(
            name=name,
            experiment_path=params.creatable_directory("experiment_output_path"),
            copy_curriculum=params.boolean("copy_curriculum", default=True),
            file_name=params.optional_string("file_name"),
            by_language_candidate_accuracy_observer=ByLanguageCandidateAccuracyObserver(
                name="yamllogger_by_language_candidate_accuracy_observer"
            )
            if params.boolean("calculate_accuracy_by_language", default=False)
            else None,
            candidate_accuracy_observer=CandidateAccuracyObserver(
                name="yamllogger_candidate_accuracy_observer",
                accuracy_to_txt=False,
            )
            if params.boolean("calculate_overall_accuracy", default=False)
            else None,
        )

    def _convert_to_output_format(
        self,
        idx: int,
        semantic_node: SemanticNode,
        linguistic_description: LinguisticDescription,
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> Mapping[str, Any]:
        return {
            "id": idx,
            "text": linguistic_description.as_token_string(),
            "confidence": f"{predicted_scene_description.description_to_confidence[linguistic_description]:.2f}",
            "type": "complex"
            if isinstance(semantic_node, (ActionSemanticNode, RelationSemanticNode))
            else "object",
            "features": sorted(
                predicted_scene_description.semantics_to_feature_strs[semantic_node]
            ),
            "sub-objects": [],
            "raw_text": None,
            "slot_alignment_to_confidence": None,
        }

    def observe(  # pylint: disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_scene_description: TopLevelLanguageLearnerDescribeReturn,
    ) -> None:
        output_dict: MutableMapping[str, Any] = dict()
        output_dict[GOLD_LANGUAGE] = true_description.as_token_string()
        output_dict[OUTPUT_LANGUAGE] = []

        experiment_dir = self.experiment_path / SITUATION_DIR_NAME.format(
            num=self.counter
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        if self.copy_curriculum and isinstance(situation, SimulationSituation):

            if isinstance(perceptual_representation, VisualPerceptionRepresentation):
                for index, image_frame in enumerate(
                    zip(perceptual_representation.frames, situation.scene_images_png)
                ):
                    image_file = Image.open(image_frame[1])
                    image_editable = ImageDraw.Draw(image_file)
                    frame_clusters = image_frame[0].clusters
                    for cluster in frame_clusters:
                        font = ImageFont.truetype(
                            str(ROBOTO_FILE),
                            14,
                        )
                        id_label = (
                            "ID: " + cluster.cluster_id[len(cluster.cluster_id) - 1]
                        )
                        font_width, font_height = font.getsize(id_label)
                        # create black highlight via a rectangle for text to ensure readability
                        image_editable.rectangle(
                            (
                                cluster.centroid_x - font_width / 2,
                                cluster.centroid_y - font_height,
                                cluster.centroid_x + font_width / 2,
                                cluster.centroid_y,
                            ),
                            fill="black",
                        )
                        image_editable.text(
                            (cluster.centroid_x, cluster.centroid_y),
                            id_label,
                            fill="white",
                            font=font,
                            anchor="ms",
                        )
                    file_name = f"id_rgb_{index}.png"
                    image_file.save(str(experiment_dir / file_name))

            for file_path in situation.all_files():
                shutil.copy(file_path, experiment_dir)

        for (
            idx,
            semantic_to_description,
        ) in enumerate(predicted_scene_description.semantics_to_descriptions.items()):
            semantic_node, linguistic_description = semantic_to_description
            output_dict[OUTPUT_LANGUAGE].append(
                self._convert_to_output_format(
                    # If the semantic node has an alignment to an original perception node in the graph (e.g. an object
                    # perception root) we want to use the original ID value to enable a matching between the produced
                    # linguistic utterance and the identified object cluster. In the case no original ID exists we want
                    # to ensure the ID is unique, so we assume all semantic nodes in the scene could have an original ID
                    # and 'count' from there. This uniqueness guarantee could be improved, but it's not a problem
                    # to potentially have non-linear unique IDs.
                    semantic_node.original_node_id  # type: ignore
                    if semantic_node.original_node_id is not None
                    else idx + len(predicted_scene_description.semantics_to_descriptions),
                    semantic_node,
                    linguistic_description,
                    predicted_scene_description,
                )
            )

        # If we have two concepts exactly, we want to generate the differences panel
        if len(output_dict[OUTPUT_LANGUAGE]) == 2:
            object_a_features = set(output_dict[OUTPUT_LANGUAGE][0]["features"])
            object_b_features = set(output_dict[OUTPUT_LANGUAGE][1]["features"])
            similar_features = sorted(object_b_features.intersection(object_a_features))
            object_a_distinct = sorted(object_a_features.difference(object_b_features))
            object_b_distinct = sorted(object_b_features.difference(object_a_features))
            output_dict["differences_panel"] = {
                f'{output_dict[OUTPUT_LANGUAGE][0]["text"]}_{output_dict[OUTPUT_LANGUAGE][0]["id"]}': [
                    line for line in object_a_distinct
                ],
                "similarities": [line for line in similar_features],
                f'{output_dict[OUTPUT_LANGUAGE][1]["text"]}_{output_dict[OUTPUT_LANGUAGE][1]["id"]}': [
                    line for line in object_b_distinct
                ],
            }

        with open(
            (experiment_dir / f"{self.file_name if self.file_name else self.name}.yaml"),
            "w",
            encoding="utf-8",
        ) as decode:
            yaml.dump(output_dict, decode)

        if self._by_language_candidate_accuracy_observer:
            self._by_language_candidate_accuracy_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        if self._candidate_accuracy_observer:
            self._candidate_accuracy_observer.observe(
                situation,
                true_description,
                perceptual_representation,
                predicted_scene_description,
            )
        self.counter += 1

    def report(self) -> None:
        if self._by_language_candidate_accuracy_observer:
            self._by_language_candidate_accuracy_observer.report()
        if self._candidate_accuracy_observer:
            self._candidate_accuracy_observer.report()


# used by TopChoiceExactMatchObserver
def _by_score(scored_description: Tuple[LinguisticDescription, float]) -> float:
    return scored_description[1]


def _by_length(scored_description: Tuple[LinguisticDescription, float]) -> int:
    return -1 * len(scored_description[0])


def pretty_descriptions(
    descriptions: Mapping[LinguisticDescription, float],
    num_descriptions: int,
    *,
    sort_by_length: bool,
) -> str:
    if len(descriptions) > 1:
        top_descriptions = take(
            num_descriptions,
            sorted(descriptions.items(), key=_by_length if sort_by_length else _by_score),
        )
        parts = ["<ul>"]
        parts.extend(
            [
                f"<li>{description.as_token_string()} ({score:.2})</li>"
                for (description, score) in top_descriptions
            ]
        )
        parts.append("</ul>")
        return "\n".join(parts)

    description = only(descriptions)
    if description is not None:
        return "".join(description.as_token_string())
    else:
        return ""
