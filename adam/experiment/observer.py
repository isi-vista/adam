import logging
from abc import ABC, abstractmethod
from typing import Generic, Mapping, Optional, Tuple

from more_itertools import only, take

from adam.curriculum_to_html import CurriculumToHtmlDumper
from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.perception import PerceptionT, PerceptualRepresentation
from adam.situation import SituationT
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.visualization.make_scenes import situation_to_filename
from attr import attrib, attrs
from attr.validators import instance_of
from vistautils.parameters import Parameters


class DescriptionObserver(Generic[SituationT, LinguisticDescriptionT, PerceptionT], ABC):
    r"""
    Something which can observe the descriptions produced by `LanguageLearner`\ s.

    Typically a `DescriptionObserver` will provide some sort of summary of its observations
    when its *report* method is called.
    """

    @abstractmethod
    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
        offset: int = 0,
    ) -> None:
        r"""
        Observe a description provided by a `LanguageLearner`.

        Args:
            situation: The `Situation` being described. This is optional.
            true_description: The "gold-standard" description of the situation.
            perceptual_representation: The `PerceptualRepresentation` of the situation received by
                                       the `LanguageLearner`.
            predicted_descriptions:  The scored `LinguisticDescription`\ s produced by
                                     the `LanguageLearner`.
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
        predicted_descriptions: Mapping[LinguisticDescription, float],
        offset: int = 0,
    ) -> None:
        self._num_observations += 1

        if predicted_descriptions:
            top_choice = max(predicted_descriptions.items(), key=_by_score)

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
    _num_predictions: int = attrib(init=False, default=0)
    _num_predictions_with_gold_in_candidates: int = attrib(init=False, default=0)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
        offset: int = 0,
    ) -> None:
        self._num_predictions += 1

        descriptions_as_token_sequences = [
            desc.as_token_sequence() for desc in predicted_descriptions
        ]
        if true_description.as_token_sequence() in descriptions_as_token_sequences:
            self._num_predictions_with_gold_in_candidates += 1

    def report(self) -> None:
        accuracy = self.accuracy()
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

    outfile_dir: str = attrib(validator=instance_of(str))
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
        output_html_path = str(logging_dir / "index.html")

        if include_links_to_images is None:
            include_links_to_images = False

        logging.info("Experiment will be logged to %s", output_html_path)

        with open(output_html_path, "w") as outfile:
            html_dumper = CurriculumToHtmlDumper()

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
            outfile_dir=output_html_path,
            html_dumper=html_dumper,
            include_links_to_images=include_links_to_images,
            num_pretty_descriptions=num_pretty_descriptions,
            sort_by_length=sort_by_length,
        )

    def pre_observer(self) -> "DescriptionObserver":  # type: ignore
        return HTMLLoggerPreObserver(name="Pre-observer", html_logger=self)

    def post_observer(self) -> "DescriptionObserver":  # type: ignore
        return HTMLLoggerPostObserver(
            name="Post-observer",
            html_logger=self,
            candidate_accuracy_observer=CandidateAccuracyObserver(
                name="Post-observer-acc"
            ),
            test_mode=False,
        )

    def test_observer(self) -> "DescriptionObserver":  # type: ignore
        return HTMLLoggerPostObserver(
            name="Test-observer",
            html_logger=self,
            candidate_accuracy_observer=CandidateAccuracyObserver(
                name="Test-observer-acc"
            ),
            test_mode=True,
        )

    def pre_observer_log(
        self,
        predicted_descriptions: Mapping[LinguisticDescription, float],
        accuracy: Optional[float] = None,
    ) -> None:
        if accuracy is not None:
            accuracy_str = f"\nAccuracy: {accuracy:2.2f}%"
        else:
            accuracy_str = ""
        self.pre_observed_description = (
            pretty_descriptions(
                predicted_descriptions,
                self._num_pretty_descriptions,
                sort_by_length=self._sort_by_length,
            )
            + accuracy_str
        )

    def post_observer_log(
        self,
        *,
        observer_name: str,
        instance_number: int,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
        test_mode: bool,
        accuracy: Optional[float],
    ):
        if accuracy is None:
            accuracy = 0.0
        learner_pre_description = self.pre_observed_description
        self.pre_observed_description = None

        learner_description = pretty_descriptions(
            predicted_descriptions,
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
        with open(self.outfile_dir, "a+") as outfile:
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
                    f"\t\t\t\t\t<td>\n"
                    f"\t\t\t\t\t<h3>Scene Renderings</h3>\n"
                    f"\t\t\t\t\t</td>\n"
                )
            outfile.write(
                f"\t\t\t</tr>\n"
                f"\t\t\t<tr>\n"
                f'\t\t\t\t<td valign="top">{situation_text}\n\t\t\t\t</td>\n'
                f'\t\t\t\t<td valign="top">{true_description_text}</td>\n'
            )
            if test_mode:
                outfile.write(
                    f'\t\t\t\t<td valign="top">{learner_description}<br/>Accuracy: {accuracy:2.2f}</td>\n'
                )
            else:
                outfile.write(
                    f'\t\t\t\t<td valign="top">{learner_pre_description}</td>\n'
                    f'\t\t\t\t<td valign="top">{learner_description}<br/>Accuracy: {accuracy:2.2f}</td>\n'
                )

            if situation and isinstance(situation, HighLevelSemanticsSituation):
                render_buttons_text = self.render_buttons_html(
                    situation, perceptual_representation
                )
            else:
                render_buttons_text = ""

            outfile.write(
                f'\t\t\t\t<td valign="top">{clickable_perception_string}\n\t\t\t\t</td>\n'
            )
            if self.include_links_to_images:
                outfile.write(f"\t\t\t\t<td valign='top'>{render_buttons_text}</td>")
            outfile.write(f"\t\t\t</tr>\n\t\t</tbody>\n\t</table>")
            outfile.write("\n</body>")

    def render_buttons_html(
        self,
        situation: HighLevelSemanticsSituation,
        perception: PerceptualRepresentation[PerceptionT],
    ) -> str:
        buttons = []
        for frame in range(3):
            filename = situation_to_filename(situation, frame)
            button_suffix = self._get_button_suffix()
            buttons.append(
                f"""
                <button onclick="myFunction('render{filename}-{button_suffix}')">View Rendering {frame + 1}</button>
                <div id="render{filename}-{button_suffix}" style="display: none">
                <img src="renders/{filename}">
                </div>
                """
            )
        if not situation.is_dynamic:
            return buttons[0]
        if perception.during and perception.during.at_some_point:
            return "".join(buttons)
        return "".join(buttons[0:2])

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

    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
        offset: int = 0,
    ) -> None:
        # pylint: disable=unused-argument
        self.html_logger.pre_observer_log(predicted_descriptions)

    def report(self) -> None:
        pass


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
    candidate_accuracy_observer = attrib(
        validator=instance_of(CandidateAccuracyObserver), kw_only=True
    )
    test_mode: bool = attrib(validator=instance_of(bool), kw_only=True)
    counter: int = attrib(kw_only=True, default=0)

    def observe(
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
        offset: int = 0,
    ) -> None:
        self.candidate_accuracy_observer.observe(
            situation, true_description, perceptual_representation, predicted_descriptions
        )
        self.html_logger.post_observer_log(
            observer_name=self.name,
            instance_number=self.counter + offset,
            situation=situation,
            true_description=true_description,
            perceptual_representation=perceptual_representation,
            predicted_descriptions=predicted_descriptions,
            test_mode=self.test_mode,
            accuracy=self.candidate_accuracy_observer.accuracy(),
        )
        self.counter += 1

    def report(self) -> None:
        pass


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
    elif len(descriptions) == 1:
        return "".join(only(descriptions).as_token_string())
    else:
        return ""
