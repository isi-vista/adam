import os
from itertools import count
from typing import List

from attr import attrs

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

    def generate(
        self,
        instances: List[InstanceGroup],
        outputdestination: str,
        title: str = "Instance Group",
        overwrite: bool = False,
    ) -> int:
        for x in range(count(instances)):
            self._generate(
                instance=instances[x],
                outputdestination=f"${outputdestination}${title}${x}.html",
                title=title + f" ${x}",
                overwrite=overwrite,
            )
        return 1

    def _generate(
        self, instance: InstanceGroup, title: str, overwrite: bool, outputdestination: str
    ) -> int:
        """

        Args:
            overwrite:
            title:

        Returns:

        """
        if os.path.isFile(outputdestination) and not overwrite:
            return 1
        html = open(outputdestination, "w")
        html.write(f"<h1>${title} - ${instance.name()}</h1>\n")
        x = 0
        for inst in instance.instances():
            if not isinstance(inst[0], HighLevelSemanticsSituation):
                raise RuntimeError(
                    f"Expected the Situation to be HighLevelSemanticsSituation got ${type(inst[0])}"
                )
            if not isinstance(inst[1], LinearizedDependencyTree):
                raise RuntimeError(
                    f"Expected the Lingustics to be LinearizedDependencyTree got ${type(inst[1])}"
                )
            if not isinstance(inst[2], PerceptualRepresentation) and not isinstance(
                inst[2].frames, DevelopmentalPrimitivePerceptionFrame
            ):
                raise RuntimeError(
                    f"Expected the Perceptual Representation to contain DevelopmentalPrimitivePerceptionFrame got ${type(inst[2].frames)}"
                )
            html.write(f"<h2>Scene ${x}</h2>\n<div>")
            html.writelines(self._situationtext(inst[0]))
            html.writelines(self._linguistictext(inst[1]))
            html.writelines(self._perceptiontext(inst[2]))
            html.write("</div>")
        html.close()

    def _situationtext(self, situation: HighLevelSemanticsSituation) -> list[str]:
        """

        Returns:

        """
        outputtext = [
            '<div><h2 id="situation">Situation Description</h2>\n',
            "<h3>Objects</h3>\n<ul>",
        ]
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
            outputtext.append(
                f"<li>${rel.relation_type.handle} with First Argument: ${rel.first_slot.ontology_node.handle} and ${rel.second_slot.ontology_node.handle}</li>"
            )
        outputtext.append(
            "</ul>\n<h3>Ontology</h3>\n<ul><li>We should probably determine some way of displaying this but I wasn't sure how?</li></ul>"
        )
        outputtext.append("</div>")
        return outputtext

    def _perceptiontext(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> list[str]:
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
                outputtext.append(
                    f"<li>${rel.relation_type.handle} for ${rel.arg1} and ${rel.arg2}"
                )
            outputtext.append("</ul>")
            x = x + 1
        outputtext.append("</div>")
        return outputtext

    def _linguistictext(self, linguistic: LinearizedDependencyTree) -> list[str]:
        """

        Returns:

        """
        outputtext = [
            '<div><h2 id="lingustics">Linguistic Description of the <a '
            'href="#situation">Situation</a></h2> ',
            " ".join(linguistic.as_token_sequence()),
            "</div>",
        ]
        return outputtext
