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
        instances: [InstanceGroup], outputdestination: str
    ) -> "CurriculumToHtml":
        """

        Args:
            instances:
            outputdestination:

        Returns:

        """
        return CurriculumToHtml(
            outputDestination=outputdestination
        )

    def _generate(self, instance: InstanceGroup, title: str, overwrite: bool) -> int:
        """

        Args:
            overwrite:
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

    def _situationtext(self, situation: HighLevelSemanticsSituation) -> list[str]:
        """

        Returns:

        """
        outputtext = ['<div><h2 id="situation">Situation Description</h2>\n', '<h3>Objects</h3>\n<ul>']
        for obj in situation.objects:
            outputtext.append(f"<li>${obj.ontology_node.handle}</li>")
            outputtext.append(f"<li>Properties:\n<ul>")
            for property in obj.properties:
                outputtext.append(f"<li>${property.handle}</li>")
            outputtext.append("</ul></li>")
        outputtext.append("</ul>\n<h3>Actions</h3>\n<ul>")
        for acts in situation.actions:
            outputtext.append(f"<li>${acts.action_type.handle}</li>")
        outputtext.append("</ul>\n<h3>Relations</h3>\n<ul>")
        for rel in situation.relations:
            outputtext.append(f"<li>${rel.relation_type.handle} with First Argument: ${rel.first_slot.ontology_node.handle} and ${rel.second_slot.ontology_node.handle}</li>")
        outputtext.append("</ul>\n<h3>Ontology</h3>\n<ul><li>We should probably determine some way of displaying this but I wasn't sure how?</li></ul>")
        outputtext.append("</div>")
        return outputtext

    def _perceptiontext(self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]) -> list[str]:
        """

        Returns:

        """
        outputtext = [
            '<div><h2 id="perception">Perception Description of the <a href="#situation">Situation</a></h2>'
        ]
        x = 0
        for frame in perception.frames:
            outputtext.append(f"<h3>Frame ${x}</h3>")
            outputtext.append("<h4>Perceived Objects</h4>\n<ul>")
            for obj in frame.perceived_objects:
                outputtext.append(f"<li>${obj.debug_handle}</li>")
            outputtext.append("</ul>\n<h4>Property Assertions</h4>\n<ul>")
            for prop in frame.property_assertions:
                outputtext.append(f"<li>${prop}")
            outputtext.append("</ul>\n<h4>Relations</h4>\n<ul>")
            for rel in frame.relations:
                outputtext.append(f"<li>${rel.relation_type.handle} for ${rel.arg1} and ${rel.arg2}")
            outputtext.append("</ul>")
            x = x+1
        outputtext.append("</div>")
        return outputtext

    def _linguistictext(self) -> list[str]:
        """

        Returns:

        """
        outputtext = ['<div><h2 id="lingustics">Linguistic Description of the <a '
                      'href="#situation">Situation</a></h2> ',
                      " ".join(self._linguistics.as_token_sequence()), "</div>"]
        return outputtext
