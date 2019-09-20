from pathlib import Path
from typing import AbstractSet, Any, Callable, Iterable, List, Tuple, TypeVar, Union

from attr import attrs
from immutablecollections import (
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutablesetmultidict,
)
from more_itertools import flatten
from networkx import DiGraph
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.preconditions import check_state

from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import PART_OF
from adam.ontology.phase1_spatial_relations import Region, SpatialPath
from adam.perception import ObjectPerception, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
)
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation

USAGE_MESSAGE = """
    curriculum_to_html.py param_file
     \twhere param_file has the following parameter:
     \t\toutput_directory: where to write the HTML output
   """


def main(params: Parameters) -> None:
    root_output_directory = params.creatable_directory("output_directory")
    phase1_curriculum_dir = root_output_directory / "gaila-phase-1"
    phase1_curriculum_dir.mkdir(parents=True, exist_ok=True)
    CurriculumToHtmlDumper().dump_to_html(
        GAILA_PHASE_1_CURRICULUM,
        output_directory=phase1_curriculum_dir,
        title="GAILA Phase 1 Curriculum",
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
        output_directory: Path,
        title: str,
    ):
        r"""
        Method to take a list of `InstanceGroup`\ s and turns each one into an individual page
        
        Given a list of `InstanceGroup`\ s and an output directory of *outputdestination*
        along with a *title* for the pages the generator loops through each group
        and calls the internal method to create HTML pages.
        """
        files_written: List[Tuple[str, str]] = []
        # write each instance group to its own file
        for (idx, instance_group) in enumerate(instance_groups):
            instance_group_header = f"{idx:03} - {instance_group.name()}"
            # not absolute because when we use this to make links in index.html,
            # we don't want them to break if the user moves the directory.
            relative_filename = f"{instance_group_header}.html"
            files_written.append((instance_group_header, relative_filename))
            CurriculumToHtmlDumper._dump_instance_group(
                self,
                instance_group=instance_group,
                output_destination=output_directory / relative_filename,
                title=f"{instance_group_header} - {title}",
            )

        # write an table of contents to index.html
        with open(output_directory / "index.html", "w") as index_out:
            index_out.write(f"<head><title>{title}</title></head><body>")
            index_out.write("<ul>")
            for (
                instance_group_title,
                instance_group_dump_file_relative_path,
            ) in files_written:
                index_out.write(
                    f"\t<li><a href='{instance_group_dump_file_relative_path}'>"
                    f"{instance_group_title}</a></li>"
                )
            index_out.write("</ul>")
            index_out.write("</body>")

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
            html_out.write(f"<head>\n\t<style>{CSS}\n\t</style>\n</head>")
            html_out.write(f"\n<body>\n\t<h1>{title} - {instance_group.name()}</h1>")
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
                    f"\n\t<table>\n"
                    f"\t\t<thead>\n"
                    f"\t\t\t<tr>\n"
                    f'\t\t\t\t<th colspan="3">\n'
                    f"\t\t\t\t\t<h2>Scene {instance_number}</h2>\n"
                    f"\t\t\t\t</th>\n\t\t\t</tr>\n"
                    f"\t\t</thead>\n"
                    f"\t\t<tbody>\n"
                    f"\t\t\t<tr>\n"
                    f"\t\t\t\t<td>\n"
                    f'\t\t\t\t\t<h3 id="situation-{instance_number}">Situation</h3>\n'
                    f"\t\t\t\t</td>\n"
                    f"\t\t\t\t<td>\n"
                    f'\t\t\t\t\t<h3 id="linguistic-{instance_number}">Language</h3>\n'
                    f"\t\t\t\t</td>\n"
                    f"\t\t\t\t<td>\n"
                    f'\t\t\t\t\t<h3 id="perception-{instance_number}">Learner Perception</h3>\n'
                    f"\t\t\t\t</td>\n"
                    f"\t\t\t</tr>\n"
                    f"\t\t\t<tr>\n"
                    f'\t\t\t\t<td valign="top">{self._situation_text(situation)}\n\t\t\t\t</td>\n'
                    f'\t\t\t\t<td valign="top">{self._linguistic_text(dependency_tree)}</td>\n'
                    f'\t\t\t\t<td valign="top">{self._perception_text(perception)}\n\t\t\t\t</td>\n'
                    f"\t\t\t</tr>\n\t\t</tbody>\n\t</table>"
                )
            html_out.write("\n</body>")

    def _situation_text(self, situation: HighLevelSemanticsSituation) -> str:
        """
        Converts a situation description into its sub-parts as a table entry
        """
        output_text = [f"\n\t\t\t\t\t<h4>Objects</h4>\n\t\t\t\t\t<ul>"]
        for obj in situation.objects:
            property_string: str
            if obj.properties:
                property_string = (
                    "[" + ",".join(prop.handle for prop in obj.properties) + "]"
                )
            else:
                property_string = ""
            output_text.append(
                f"\t\t\t\t\t\t<li>{obj.ontology_node.handle}{property_string}</li>"
            )
        output_text.append("\t\t\t\t\t</ul>")
        if situation.actions:
            output_text.append("\t\t\t\t\t<h4>Actions</h4>\n\t\t\t\t\t<ul>")
            for acts in situation.actions:
                output_text.append(f"\t\t\t\t\t\t<li>{acts.action_type.handle}</li>")
            output_text.append("\t\t\t\t\t</ul>")
        if situation.always_relations:
            output_text.append("\t\t\t\t\t<h4>Relations</h4>\n\t\t\t\t\t<ul>")
            for rel in situation.always_relations:
                output_text.append(
                    f"\t\t\t\t\t\t<li>{rel.relation_type.handle}({rel.first_slot.ontology_node.handle},"
                    f"{self._situation_object_or_region_text(rel.second_slot)})</li>"
                )
            output_text.append("\t\t\t\t\t</ul>")
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

        check_state(
            len(perception.frames) in (1, 2),
            "Only know how to handle 1 or 2 frame " "perceptions for now",
        )

        perception_is_dynamic = len(perception.frames) > 1

        # first, we build an index of objects to their properties.
        # This will be used so that when we list the objects,
        # we can easily list their properties in brackets right after them.
        def extract_subject(prop: PropertyPerception) -> ObjectPerception:
            return prop.perceived_object

        first_frame_properties = _index_to_setmultidict(
            perception.frames[0].property_assertions, extract_subject
        )
        second_frame_properties = (
            _index_to_setmultidict(
                perception.frames[1].property_assertions, extract_subject
            )
            if perception_is_dynamic
            else immutablesetmultidict()
        )

        # Next, we determine what objects persist between both frames
        # and which do not.
        first_frame_objects = perception.frames[0].perceived_objects
        second_frame_objects = (
            perception.frames[1].perceived_objects
            if perception_is_dynamic
            else immutableset()
        )
        static_objects = (
            first_frame_objects.intersection(second_frame_objects)
            if perception_is_dynamic
            else first_frame_objects
        )
        all_objects = first_frame_objects.union(second_frame_objects)

        # For objects, properties, and relations we will use arrows to indicate
        # when something beings or ceased to exist between frames.
        # Since the logic will be the same for all three types,
        # we pull it out into a function.
        def compute_arrow(
            item: Any, static_items: AbstractSet[Any], first_frame_items: AbstractSet[Any]
        ) -> Tuple[str, str]:
            if item in static_items:
                # item doesn't change - no arrow
                return ("", "")
            elif item in first_frame_items:
                # item ceases to exist
                return ("", " ---> Ø")
            else:
                # item beings to exist in the second frame
                return ("Ø ---> ", "")

        # the logic for rendering objects, which will be used in the loop below.
        # This needs to be an inner function so it can access the frame property maps, etc.
        def render_object(obj: ObjectPerception) -> str:
            obj_text = f"<i>{obj.debug_handle}</i>"
            first_frame_obj_properties = first_frame_properties[obj]
            second_frame_obj_properties = second_frame_properties[obj]
            static_properties = (
                second_frame_obj_properties.intersection(first_frame_obj_properties)
                if second_frame_obj_properties
                else first_frame_obj_properties
            )

            # logic for rendering properties, for use in the loop below.
            def render_property(prop: PropertyPerception) -> str:
                (prop_prefix, prop_suffix) = compute_arrow(
                    prop, static_properties, first_frame_obj_properties
                )
                prop_string: str
                if isinstance(prop, HasColor):
                    prop_string = (
                        f'<span style="background-color: {prop.color}; '
                        f'color: {prop.color}; border: 1px solid black;">Object Color</span>'
                    )
                elif isinstance(prop, HasBinaryProperty):
                    prop_string = str(prop.binary_property)
                else:
                    raise RuntimeError(f"Cannot render property: {prop}")

                return f"{prop_prefix}{prop_string}{prop_suffix}"

            all_properties: ImmutableSet[PropertyPerception] = immutableset(
                flatten([first_frame_obj_properties, second_frame_obj_properties])
            )
            prop_strings = [render_property(prop) for prop in all_properties]

            if prop_strings:
                return f"{obj_text}[{'; '.join(prop_strings)}]"
            else:
                return obj_text

        # Here we process the relations between the two scenes to determine all relations.
        # This has to be done before rending objects so we can use the PART_OF relation to order
        # the objects.
        first_frame_relations = perception.frames[0].relations
        second_frame_relations = (
            perception.frames[1].relations if perception_is_dynamic else immutableset()
        )
        static_relations = (
            second_frame_relations.intersection(first_frame_relations)
            if perception_is_dynamic
            else first_frame_relations
        )
        all_relations = first_frame_relations.union(second_frame_relations)

        # Here we add the perceived objects to a NetworkX DiGraph with PART_OF relations being the
        # edges between objects. This allows us to do pre-order traversal of the Graph to make a
        # nested <ul></ul> for the objects rather than a flat list.
        graph = DiGraph()
        root = ObjectPerception("root")
        graph.add_node(root)
        expressed_relations = set()

        for object_ in all_objects:
            graph.add_node(object_)
            graph.add_edge(root, object_)

        for relation_ in all_relations:
            if relation_.relation_type == PART_OF:
                graph.add_edge(relation_.first_slot, relation_.second_slot)
                expressed_relations.add(relation_)

        # Next, we render objects, together with their properties, using preorder DFS Traversal
        # We also add in `In Region` relationships at this step for objects which have them.
        output_text.append("\n\t\t\t\t\t<h5>Perceived Objects</h5>\n\t\t\t\t\t<ul>")
        visited = set()
        region_relations = immutableset(
            region for region in all_relations if region.relation_type == IN_REGION
        )

        # This loop doesn't quite get the tab spacing right. It could at the cost of increased
        # complexity. Would need to track the "depth" we are currently at.
        def dfs_walk(node):
            visited.add(node)
            if not node == root:
                (obj_prefix, obj_suffix) = compute_arrow(
                    node, static_objects, first_frame_objects
                )
                output_text.append(
                    f"\t\t\t\t\t\t<li>{obj_prefix}{render_object(node)}{obj_suffix}<ul>"
                )
                # Handle Region Relations
                for region_relation in region_relations:
                    if region_relation.first_slot == node:
                        (relation_prefix, relation_suffix) = compute_arrow(
                            region_relation, static_relations, first_frame_relations
                        )
                        output_text.append(
                            f"\t\t\t\t\t\t<li>{relation_prefix}{region_relation}{relation_suffix}</li>"
                        )
                        expressed_relations.add(region_relation)
            for succ in graph.successors(node):
                if succ not in visited:
                    dfs_walk(succ)
            output_text.append("\t\t\t\t\t\t</ul></li>")

        dfs_walk(root)
        output_text.append("\t\t\t\t\t</ul>")

        # Finally we render all relations between objects
        if all_relations:
            output_text.append("\t\t\t\t\t<h5>Other Relations</h5>\n\t\t\t\t\t<ul>")

            for relation in all_relations:
                if relation not in expressed_relations:
                    (relation_prefix, relation_suffix) = compute_arrow(
                        relation, static_relations, first_frame_relations
                    )
                    output_text.append(
                        f"\t\t\t\t\t\t<li>{relation_prefix}{relation}{relation_suffix}</li>"
                    )
            output_text.append("\t\t\t\t\t</ul>")

        if perception.during:
            output_text.append("\t\t\t\t\t<h5>During the action</h5>\n\t\t\t\t\t<ul>")
            output_text.append(self._render_during(perception.during, indent_depth=5))

        return "\n".join(output_text)

    def _render_during(
        self, during: DuringAction[ObjectPerception], *, indent_depth: int = 0
    ) -> str:
        indent = "\t" * indent_depth
        lines = [f"{indent}<ul>"]
        if during.objects_to_paths:
            lines.append(f"{indent}\t<li><b>Paths:</b>")
            lines.append(f"{indent}\t<ul>")
            for (object_, path) in during.objects_to_paths.items():
                path_rendering = self._render_path(path, indent_depth=indent_depth + 2)
                lines.append(f"{indent}\t\t<li>{object_}: {path_rendering}</li>")
            lines.append(f"{indent}</ul></li>")
        if during.continuously:
            lines.append(f"{indent}\t<li><b>Relations which hold continuously:</b>")
            lines.append(f"{indent}\t<ul>")
            for relation in during.continuously:
                lines.append(f"{indent}\t\t<li>{relation}</li>")
            lines.append(f"{indent}</ul></li>")
        if during.at_some_point:
            lines.append(f"{indent}\t<li><b>Relations which hold at some point:</b>")
            lines.append(f"{indent}\t<ul>")
            for relation in during.at_some_point:
                lines.append(f"{indent}\t\t<li>{relation}</li>")
            lines.append(f"{indent}</ul></li>")

        return "\n".join(lines)

    def _render_path(
        self, path: SpatialPath[ObjectPerception], *, indent_depth: int = 0
    ) -> str:
        indent = "\t" * indent_depth
        lines = [f"{indent}<ul>"]
        lines.append(f"{indent}\t<li>")
        lines.append(str(path))
        lines.append(f"{indent}\t</li>")
        return "\n".join(lines)

    def _linguistic_text(self, linguistic: LinearizedDependencyTree) -> str:
        """
        Parses the Linguistic Description of a Linearized Dependency Tree into a table entry

        Takes a `LinearizedDependencyTree` which is turned into a token sequence and
        phrased as a sentence for display. Returns a List[str]
        """
        return " ".join(linguistic.as_token_sequence())


CSS = """
body {
    font-size: 1em;
    font-family: sans-serif;
}

table td { 
    padding: 1em; 
    background-color: #FAE5D3 ;
}
"""

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _index_to_setmultidict(
    items: Iterable[_VT], index_func: Callable[[_VT], _KT]
) -> ImmutableSetMultiDict[_KT, _VT]:
    return immutablesetmultidict((index_func(x), x) for x in items)


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
