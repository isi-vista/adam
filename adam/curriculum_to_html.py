import os

from attr import attrs, attrib
from attr.validators import instance_of

from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class CurriculumToHtml:
    """

    """

    _situation: HighLevelSemanticsSituation = attrib(
        instance_of(HighLevelSemanticsSituation)
    )
    _linguistics: LinearizedDependencyTree = attrib(instance_of(LinearizedDependencyTree))
    _perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame] = attrib(
        instance_of(PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame])
    )
    _outputDestination: str = attrib(instance_of(str))

    @staticmethod
    def createbuilder(
        instance: InstanceGroup, outputdestination: str
    ) -> "CurriculumToHtml":
        """

        Args:
            instance:
            outputdestination:

        Returns:

        """
        instances = instance.instances()
        return CurriculumToHtml(
            situation=instances[0],
            linguistics=instances[1],
            perception=instances[2],
            outputDestination=outputdestination,
        )

    def generate(self, title: str, overwrite: bool) -> int:
        """

        Args:
            title:

        Returns:

        """
        if os.path.isFile(self._outputDestination) and not overwrite:
            return 1
        html = open(self._outputDestination, "w")
        html.write(f"<h1>${title}</h1>\n")
        html.writelines(self._situationtext())
        html.writelines(self._linguistictext())
        html.writelines(self._perceptiontext())
        html.close()

    def _situationtext(self) -> list[str]:
        """

        Returns:

        """
        outputtext = ['<div><h2 id="situation">Situation Description']

        outputtext.append("</div>")
        return outputtext

    def _perceptiontext(self) -> list[str]:
        """

        Returns:

        """
        outputtext = [
            '<div><h2 id="perception">Perception Description of the <a href="#situation">Situation</a>'
        ]

        outputtext.append("</div>")
        return outputtext

    def _linguistictext(self) -> list[str]:
        """

        Returns:

        """
        outputtext = [
            '<div><h2 id="lingustics">Linguistic Description of the <a href="#situation">Situation</a>'
        ]

        outputtext.append("</div>")
        return outputtext
