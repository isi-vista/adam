from pathlib import Path
from typing import Iterable, Union, List

from attr import attrs
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.preconditions import check_state

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology.phase1_spatial_relations import Region
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasColor,
)
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


USAGE_MESSAGE = """
    curriculum_to_html.py param_file
     \twhere param_file has the following parameter:
     \t\toutput_directory: where to write the HTML output
   """


def main(params: Parameters) -> None:
    curriculum_dumper = CurriculumToHtmlDumper()
    curriculum_dumper.dump_to_html(
        GAILA_PHASE_1_CURRICULUM,
        output_destination=params.creatable_directory("output_directory"),
        title="GAILA PHASE 1 CURRICULUM",
    )


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
            for (instance_number, (situation, dependency_tree, perception)) in enumerate(
                instance_group.instances()
            ):
                if not isinstance(situation, HighLevelSemanticsSituation):
                    raise RuntimeError(
                        f"Expected the Situation to be HighLevelSemanticsSituation got {type(situation)}"
                    )
                if not isinstance(dependency_tree, LinearizedDependencyTree):
                    raise RuntimeError(
                        f"Expected the Lingustics to be LinearizedDependencyTree got {type(dependency_tree)}"
                    )
                if not (
                    isinstance(perception, PerceptualRepresentation)
                    and isinstance(
                        perception.frames[0], DevelopmentalPrimitivePerceptionFrame
                    )
                ):
                    raise RuntimeError(
                        f"Expected the Perceptual Representation to contain DevelopmentalPrimitivePerceptionFrame got "
                        f"{type(perception.frames)}"
                    )

                html_out.write(
                    f"<table>\n"
                    f"\t<thead>\n"
                    f"\t\t<tr>\n"
                    f'\t\t\t<th colspan="3">\n'
                    f"\t\t\t<h2>Scene {instance_number}</h2>\n"
                    f"\t\t</th>\n\t</tr>\n"
                    f"</thead>\n"
                    f"<tbody>\n"
                    f"\t<tr>\n"
                    f"\t\t<td>\n"
                    f'\t\t\t<h3 id="situation-{instance_number}">Situation</h3>\n'
                    f"\t\t</td>\n"
                    f"\t\t<td>\n"
                    f'\t\t\t<h3 id="linguistic-{instance_number}">Language</h3>\n'
                    f"\t\t</td>\n"
                    f"\t\t<td>\n"
                    f'\t\t\t<h3 id="perception-{instance_number}">Learner Perception</h3>\n'
                    f"\t\t</td>\n"
                    f"\t</tr>\n"
                    f"\t<tr>\n"
                    f'\t\t<td valign="top">{self._situation_text(situation)}</td>\n'
                    f'\t\t<td valign="top">{self._linguistic_text(dependency_tree)}</td>\n'
                    f'\t\t<td valign="top">{self._perception_text(perception)}</td>\n'
                    f"</tr></tbody>"
                )

    def _situation_text(self, situation: HighLevelSemanticsSituation) -> str:
        """
        Converts a situation description into its sub-parts as a table entry
        """
        output_text = [f"\t\t\t<h4>Objects</h4>\n\t\t\t<ul>"]
        for obj in situation.objects:
            property_string: str
            if obj.properties:
                property_string = (
                    "[" + ",".join(prop.handle for prop in obj.properties) + "]"
                )
            else:
                property_string = ""
            output_text.append(
                f"\t\t\t\t<li>{obj.ontology_node.handle}{property_string}</li>"
            )
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
        return "\n".join(output_text)

    def _situation_object_or_region_text(
        self, obj_or_region: Union[SituationObject, Region[SituationObject]]
    ) -> str:
        if isinstance(obj_or_region, SituationObject):
            return obj_or_region.ontology_node.handle
        else:
            return str(obj_or_region)

    def _perception_text(
        self, perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ) -> str:
        """
        Turns a perception into a list of items in the perceptions frames.
        """
        output_text: List[str] = []
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
                    if isinstance(prop, HasColor):
                        output_text.append(
                            f'\t\t\t\t\t<li>{prop} - <span style="background-color: {prop.color}; '
                            f'color: {prop.color}; border: 1px solid black;">Object Color</span></li>'
                        )
                    else:
                        output_text.append(f"\t\t\t\t\t<li>{prop}</li>")
                output_text.append("\t\t\t\t</ul>")
            if frame.relations:
                output_text.append("\t\t\t\t<h5>Relations</h5>\n\t\t\t\t<ul>")
                for rel in frame.relations:
                    output_text.append(
                        f"\t\t\t\t\t<li>{rel.relation_type.handle}({rel.first_slot},{rel.second_slot})"
                    )
                output_text.append("\t\t\t\t</ul>")
            frame_number = frame_number + 1
        return "\n".join(output_text)

    def _linguistic_text(self, linguistic: LinearizedDependencyTree) -> str:
        """
        Parses the Linguistic Description of a Linearized Dependency Tree into a table entry

        Takes a `LinearizedDependencyTree` which is turned into a token sequence and
        phrased as a sentence for display. Returns a List[str]
        """
        return " ".join(linguistic.as_token_sequence())


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
