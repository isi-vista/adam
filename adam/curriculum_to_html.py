from pathlib import Path
from typing import Iterable, Union

from attr import attrs
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology import Region
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class CurriculumToHtmlDumper:
    """
    Class to turn an `InstanceGroup` into an html document
    """

    def dump_to_html(
        self,
        instance_groups: Iterable[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
        *,
        output_destination: Path,
        title: str = "Instance Group",
    ):
        r"""
        Method to take a list of `InstanceGroup`\ s and turns each one into an individual page
        
        Given a list of InstanceGroups and an output directory of *outputdestination* along with 
        a *title* for the pages the generator loops through each group and calls the internal 
        method to create HTML pages.
        """
        for (idx, instance_group) in enumerate(instance_groups):
            CurriculumToHtmlDumper._dump_instance_group(
                self,
                instance_group=instance_group,
                output_destination=output_destination.joinpath(f"{title}{idx}.html"),
                title=f"{title} {idx}",
            )

    def _dump_instance_group(
        self,
        instance_group: InstanceGroup[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
            DevelopmentalPrimitivePerceptionFrame,
        ],
        title: str,
        output_destination: Path,
    ):
        """
        Internal generation method for individual instance groups into HTML pages

        Given an `InstanceGroup` with a `HighLevelSemanticsSituation`,
        `LinearizedDependencyTree`, and `DevelopmentalPrimitivePerceptionFrame` this function
        creates an html page at the given *outputdestination* and *title*. If the file already
        exists and *overwrite* is set to False an error is raised in execution. Each page turns an
        instance group with each "instance" as an indiviudal section on the page.
        """
        with open(output_destination, "w") as html_out:
            html_out.write(f"<h1>{title} - {instance_group.name()}</h1>\n")
            for (instance_number, inst) in enumerate(instance_group.instances()):
                if not isinstance(inst[0], HighLevelSemanticsSituation):
                    raise RuntimeError(
                        f"Expected the Situation to be HighLevelSemanticsSituation got {type(inst[0])}"
                    )
                if not isinstance(inst[1], LinearizedDependencyTree):
                    raise RuntimeError(
                        f"Expected the Lingustics to be LinearizedDependencyTree got {type(inst[1])}"
                    )
                if not (
                    isinstance(inst[2], PerceptualRepresentation)
                    and isinstance(
                        inst[2].frames[0], DevelopmentalPrimitivePerceptionFrame
                    )
                ):
                    raise RuntimeError(
                        f"Expected the Perceptual Representation to contain DevelopmentalPrimitivePerceptionFrame got {type(inst[2].frames)}"
                    )
                html_out.write(
                    f'<table>\n<thead>\n\t<tr>\n\t\t<th colspan="3">\n\t\t\t<h2>Scene {instance_number}</h2>\n\t\t</th>\n\t</tr>\n</thead>\n'
                    f'<tbody>\n\t<tr>\n\t\t<td>\n\t\t\t<h3 id="situation-{instance_number}">Situation Description</h3>\n\t\t</td>\n\t\t'
                    f'<td>\n\t\t\t<h3 id="lingustics-{instance_number}">Linguistic Descrption</h3>\n\t\t</td>\n\t\t'
                    f'<td>\n\t\t\t<h3 id="perception-{instance_number}">Perception Description</h3>\n\t\t</td>\n\t</tr>\n\t<tr>'
                )
                html_out.writelines(CurriculumToHtmlDumper._situation_text(self, inst[0]))
                html_out.writelines(
                    CurriculumToHtmlDumper._linguistic_text(self, inst[1])
                )
                html_out.writelines(
                    CurriculumToHtmlDumper._perception_text(self, inst[2])
                )
                html_out.write("</tr></tbody>")

    def _situation_text(self, situation: HighLevelSemanticsSituation) -> Iterable[str]:
        """
        Converts a situation description into its sub-parts as a table entry
        """
        output_text = [f"\t\t<td>\n\t\t\t<h4>Objects</h4>\n\t\t\t<ul>"]
        for obj in situation.objects:
            output_text.append(f"\t\t\t\t<li>{obj.ontology_node.handle}</li>")
            # Reintroduce Properties as [] around the object
            # output_text.append(f"<li>Properties:\n<ul>")
            # for prop in obj.properties:
            #    output_text.append(f"<li>{prop.handle}</li>")
            # output_text.append("</ul></li>")
        output_text.append("\t\t\t</ul>")
        if situation.actions:
            output_text.append("\t\t\t<h4>Actions</h4>\n\t\t\t\t<ul>")
            for acts in situation.actions:
                output_text.append(f"\t\t\t\t<li>{acts.action_type.handle}</li>")
            output_text.append("\t\t\t</ul>")
        if situation.relations:
            output_text.append("\t\t\t<h4>Relations</h4>\n\t\t\t<ul>")
            for rel in situation.relations:
                output_text.append(
                    f"\t\t\t\t<li>{rel.relation_type.handle}({rel.first_slot.ontology_node.handle},"
                    f"{self._situation_object_or_region_text(rel.second_slot)})</li>"
                )
            output_text.append("\t\t\t</ul>")
        output_text.append("\t\t</td>")
        return output_text

    def _situation_object_or_region_text(
        self, obj_or_region: Union[SituationObject, Region[SituationObject]]
    ) -> str:
        if isinstance(obj_or_region, SituationObject):
            return obj_or_region.ontology_node.handle
        else:
            return str(obj_or_region)

    def _perception_text(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> Iterable[str]:
        """
        Turns a perception into a list of items in the perceptions frames.
        """
        output_text = [f"\t\t<td>"]
        frame_number = 0
        for frame in perception.frames:
            output_text.append(f"\t\t\t<h4>Frame {frame_number}</h4>")
            output_text.append("\t\t\t\t<h5>Perceived Objects</h5>\n\t\t\t\t<ul>")
            for obj in frame.perceived_objects:
                output_text.append(f"\t\t\t\t\t<li>{obj.debug_handle}</li>")
            output_text.append("\t\t\t\t</ul>")
            if frame.property_assertions:
                output_text.append("\t\t\t\t<h5>Property Assertions</h5>\n\t\t\t\t<ul>")
                for prop in frame.property_assertions:
                    output_text.append(f"\t\t\t\t\t<li>{prop}")
                output_text.append("\t\t\t\t</ul>")
            if frame.relations:
                output_text.append("\t\t\t\t<h5>Relations</h5>\n\t\t\t\t<ul>")
                for rel in frame.relations:
                    output_text.append(
                        f"\t\t\t\t\t<li>{rel.relation_type.handle}({rel.arg1},{rel.arg2})"
                    )
                output_text.append("\t\t\t\t</ul>")
            frame_number = frame_number + 1
        output_text.append("\t\t</td>")
        return output_text

    def _linguistic_text(self, linguistic: LinearizedDependencyTree) -> Iterable[str]:
        """
        Parses the Linguistic Description of a Linearized Dependency Tree into a table entry

        Takes a `LinearizedDependencyTree` which is turned into a token sequence and
        phrased as a sentence for display. Returns a List[str]
        """
        return [f"\t\t<td>", " ".join(linguistic.as_token_sequence()), "\t\t</td>"]


def main(params: Parameters) -> None:
    curriculum_dumper = CurriculumToHtmlDumper()
    curriculum_dumper.dump_to_html(
        GAILA_PHASE_1_CURRICULUM,
        output_destination=params.creatable_directory("output_directory"),
        title="GAILA PHASE 1 CURRICULUM",
    )


USAGE_MESSAGE = """
    curriculum_to_html.py param_file
     \twhere param_file has the following parameter:
     \t\toutput_directory: where to write the HTML output
   """

if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
