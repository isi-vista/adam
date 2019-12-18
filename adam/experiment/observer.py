import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Mapping, Optional, Tuple, io

from attr import attrib, attrs
from attr.validators import instance_of

from adam.curriculum_to_html import CurriculumToHtmlDumper
from adam.language import LinguisticDescription, LinguisticDescriptionT
from adam.language.language_generator import SituationT
from adam.perception import PerceptionT, PerceptualRepresentation


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
class HTMLLogger:

    output_dir: Path = attrib(validator=instance_of(Path), kw_only=True)
    experiment_name: str = attrib(validator=instance_of(str), kw_only=True)
    curriculum_name: str = attrib(validator=instance_of(str), kw_only=True)
    logging_dir: Path = attrib(init=False)
    outfile: io = attrib(init=False)
    html_dumper: CurriculumToHtmlDumper = attrib(init=False)

    def __attrs_post_init__(self):
        self.logging_dir = self.output_dir / self.experiment_name
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.outfile = open(self.logging_dir / (self.curriculum_name + '.html'), 'w')
        self.html_dumper = CurriculumToHtmlDumper()
        self.outfile.write(f"<head>\n\t<style>{CSS}\n\t</style>\n</head>")
        self.outfile.write(f"\n<body>\n\t<h1>{self.experiment_name} - {self.curriculum_name}</h1>")
        self.outfile.write("""
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


    def log(self, *, instance_number: int, observer_name: str, true_description: str, learner_description: str, situation_string: str, perception_string: str):

        clickable_perception_string = f"""
            <button onclick="myFunction('myPerception{instance_number}')">View Perception</button>
            <div id="myPerception{instance_number}" style="display: none">
            {perception_string}
            </div>
            """

        self.outfile.write(
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
            f'\t\t\t\t\t<h3 id="learner-{instance_number}">Learner\'s Description</h3>\n'
            f"\t\t\t\t</td>\n"
            f"\t\t\t\t<td>\n"
            f'\t\t\t\t\t<h3 id="perception-{instance_number}">Learner Perception</h3>\n'
            f"\t\t\t\t</td>\n"
            f"\t\t\t</tr>\n"
            f"\t\t\t<tr>\n"
            f'\t\t\t\t<td valign="top">{situation_string}\n\t\t\t\t</td>\n'
            f'\t\t\t\t<td valign="top">{true_description}</td>\n'
            f'\t\t\t\t<td valign="top">{learner_description}</td>\n'
            f'\t\t\t\t<td valign="top">{clickable_perception_string}\n\t\t\t\t</td>\n'
            f"\t\t\t</tr>\n\t\t</tbody>\n\t</table>"
        )
        self.outfile.write("\n</body>")


@attrs(slots=True)
class HTMLLoggerObserver(DescriptionObserver[SituationT, LinguisticDescriptionT, PerceptionT]):
    r"""
    Logs the true description and learner's descriptions throughout the learning process.
    """
    name: str = attrib(validator=instance_of(str))
    html_logger: HTMLLogger = attrib(init=True, validator=instance_of(HTMLLogger), kw_only=True)
    counter: int = attrib(kw_only=True, default=0)

    def observe(  # pylint:disable=unused-argument
        self,
        situation: Optional[SituationT],
        true_description: LinguisticDescription,
        perceptual_representation: PerceptualRepresentation[PerceptionT],
        predicted_descriptions: Mapping[LinguisticDescription, float],
    ) -> None:

        # Convert the data to html text
        top_choice = ''
        if predicted_descriptions:
            description: LinguisticDescription = max(predicted_descriptions.items(), key=_by_score)[0]
            top_choice = ' '.join(description.as_token_sequence())

        situation_text, _ = self.html_logger.html_dumper._situation_text(situation)
        perception_text = self.html_logger.html_dumper._perception_text(perceptual_representation)
        true_description_text = ' '.join(true_description.as_token_sequence())

        # Log into html file
        # We want to log the true description, the learners guess, the perception, and situation
        self.html_logger.log(
            instance_number=self.counter,
            observer_name=self.name,
            true_description=true_description_text,
            learner_description=top_choice,
            situation_string=situation_text,
            perception_string=perception_text
        )
        self.counter += 1

    def report(self) -> None:
        pass


# used by TopChoiceExactMatchObserver
def _by_score(scored_description: Tuple[LinguisticDescription, float]) -> float:
    return scored_description[1]
