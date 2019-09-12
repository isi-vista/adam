import os
from typing import List

from attr import attrs

from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class CurriculumToHtml:
    """
    Class to turn an `InstanceGroup` into an html document
    """

    @staticmethod
    def generate(
        instances: List[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
        outputdestination: str,
        title: str = "Instance Group",
        overwrite: bool = False,
    ) -> int:
        for x in range(len(instances)):  # pylint:disable=consider-using-enumerate
            CurriculumToHtml._generate(
                instance=instances[x],
                outputdestination=f"{outputdestination}{title}{x}.html",
                title=title + f" {x}",
                overwrite=overwrite,
            )
        return 1

    @staticmethod
    def _generate(
        instance: InstanceGroup[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
            DevelopmentalPrimitivePerceptionFrame,
        ],
        title: str,
        overwrite: bool,
        outputdestination: str,
    ) -> int:
        """

        Args:
            overwrite:
            title:

        Returns:

        """
        if os.path.isfile(outputdestination) and not overwrite:
            return 1
        html = open(outputdestination, "w")
        html.write(f"<h1>{title} - {instance.name()}</h1>\n")
        instance_number = 0
        for inst in instance.instances():
            if not isinstance(inst[0], HighLevelSemanticsSituation):
                raise RuntimeError(
                    f"Expected the Situation to be HighLevelSemanticsSituation got {type(inst[0])}"
                )
            if not isinstance(inst[1], LinearizedDependencyTree):
                raise RuntimeError(
                    f"Expected the Lingustics to be LinearizedDependencyTree got {type(inst[1])}"
                )
            if not isinstance(inst[2], PerceptualRepresentation) and not isinstance(
                inst[2].frames, DevelopmentalPrimitivePerceptionFrame
            ):
                raise RuntimeError(
                    f"Expected the Perceptual Representation to contain DevelopmentalPrimitivePerceptionFrame got {type(inst[2].frames)}"
                )
            html.write(
                f'<table>\n<thead><tr><th colspan="3"><h2>Scene {instance_number}</h2>\n</th></tr></thead>'
                f'<tbody><tr><td><h3 id="situation{instance_number}">Situation Description</h3></td>'
                f'<td><h3 id="lingustics{instance_number}">Linguistic Descrption</h3></td>'
                f'<td><h3 id="perception{instance_number}">Perception Description</h3></td></tr>\n<tr>'
            )
            html.writelines(CurriculumToHtml._situationtext(inst[0]))
            html.writelines(CurriculumToHtml._linguistictext(inst[1]))
            html.writelines(CurriculumToHtml._perceptiontext(inst[2]))
            html.write("</tr></tbody>")
            instance_number = instance_number + 1
        html.close()
        return 0

    @staticmethod
    def _situationtext(situation: HighLevelSemanticsSituation):
        """

        Returns:

        """
        outputtext = [f"<td>\n", "<h4>Objects</h4>\n<ul>"]
        for obj in situation.objects:
            outputtext.append(f"<li>{obj.ontology_node.handle}</li>")
            # Reintroduce Properties as [] around the object
            # outputtext.append(f"<li>Properties:\n<ul>")
            # for prop in obj.properties:
            #    outputtext.append(f"<li>{prop.handle}</li>")
            # outputtext.append("</ul></li>")
        outputtext.append("</ul>")
        if not situation.actions:
            outputtext.append("<h4>Actions</h4>\n<ul>")
            for acts in situation.actions:
                outputtext.append(f"<li>{acts.action_type.handle}</li>")
            outputtext.append("</ul>")
        if not situation.relations:
            outputtext.append("<h4>Relations</h4>\n<ul>")
            for rel in situation.relations:
                outputtext.append(
                    f"<li>{rel.relation_type.handle}({rel.first_slot.ontology_node.handle},{rel.second_slot.ontology_node.handle})</li>"
                )
            outputtext.append("</ul>")
        outputtext.append("</td>")
        return outputtext

    @staticmethod
    def _perceptiontext(
        perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ):
        """

        Returns:

        """
        outputtext = [f"<td>"]
        frame_number = 0
        for frame in perception.frames:
            outputtext.append(f"<h4>Frame {frame_number}</h4>")
            outputtext.append("<h5>Perceived Objects</h5>\n<ul>")
            for obj in frame.perceived_objects:
                outputtext.append(f"<li>{obj.debug_handle}</li>")
            outputtext.append("</ul>")
            if not frame.property_assertions:
                outputtext.append("<h5>Property Assertions</h5>\n<ul>")
                for prop in frame.property_assertions:
                    outputtext.append(f"<li>{prop}")
                outputtext.append("</ul>")
            if not frame.relations:
                outputtext.append("<h5>Relations</h5>\n<ul>")
                for rel in frame.relations:
                    outputtext.append(
                        f"<li>{rel.relation_type.handle}({rel.arg1},{rel.arg2})"
                    )
                outputtext.append("</ul>")
            frame_number = frame_number + 1
        outputtext.append("</td>")
        return outputtext

    @staticmethod
    def _linguistictext(linguistic: LinearizedDependencyTree):
        """

        Returns:

        """
        outputtext = [f"<td>", " ".join(linguistic.as_token_sequence()), "</td>"]
        return outputtext
