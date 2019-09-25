import shutil
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    List,
    Tuple,
    TypeVar,
    Union,
    Optional,
)

from attr import attrib, attrs
from attr.validators import instance_of
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
from adam.geon import Geon
from adam.axes import WORLD_AXES, AxesInfo
from adam.language.dependency import LinearizedDependencyTree
from adam.ontology import IN_REGION
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import PART_OF, IS_SPEAKER, SMALLER_THAN, BIGGER_THAN
from adam.ontology.phase1_spatial_relations import Region, SpatialPath
from adam.perception import ObjectPerception, PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
)
from adam.relation import Relation
from adam.situation import SituationObject, SituationRegion
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.utilities import sign
import random

USAGE_MESSAGE = """
    curriculum_to_html.py param_file
     \twhere param_file has the following parameter:
     \t\toutput_directory: where to write the HTML output
   """

EXPLANATION_HEADER = (
    "\n<h2>How to read</h2>"
    "\n<p>A Situation is curriculum-designer-facing; it is not accessible to the learner. "
    "The name of a perceived object is derived from the type of the corresponding SituationObject, and the "
    "number indicates which object of its type it is - e.g. if there are two babies in a situation, "
    "the perceived objects will be 'person_0' and 'person_1'. "
    "Names on Perceptions are handles for debugging; they are not accessible to the "
    "learner. 'GAZED-AT' refers the object of focus of the speaker.</p> "
    "\n<p>Properties</p>\n<ul>"
    "\n\t<li>These define characteristics of an object that may influence perception and fulfillment of "
    "semantic roles.</li> "
    "\n\t<li>Meta-properties provide information about a property, e.g. whether or not a property can "
    "be perceived by the learner.</li> "
    "\n\t<li>OBJECT[PROPERTY_0[META_PROPERTY], PROPERTY_1, PROPERTY_2]</li> "
    "\n</ul>\n<ul>\n<p>Relations</p> "
    "\n\t<li>These define relations between two objects; e.g. smallerThan(A, B) indicates that object A is "
    "smaller than object B.</li> "
    "\n\t<li>IN_REGION is a special relation that holds between a SituationObject and a Region. These "
    "relations can be nested, and for readability, they do not fall under 'Other Relations'."
    "\n\t<li>RELATION(OBJECT_0, OBJECT_1)</li>"
    "\n</ul>\n<ul>\n<p>Arrows</p>"
    "\n\t<li>A relation preceded by 'Ø --->' is one that begins to exist by the end of the situation.</li>"
    "\n\t<li>A relation followed by '---> Ø' is one that ceases to exist by the end of the situation.</li>"
    "\n\t<li>A relation without an arrow holds true through the duration of the situation.</li>"
    "\n</ul>\n<ul>\n<p>Part-of nesting</p>"
    "\n\t<li>PART_OF relations indicate that an object is a part of another. Objects that form a "
    "structural schema can be represented in terms of several part-of relations.</li> "
    "\n\t<li>e.g. a finger is a part of a hand which is part of an arm which is part of a body</li>"
    "\n</ul>\n<ul>\n<p>Regions</p> "
    "\n\t<li>These represent regions of space. They are defined in terms of distance and/or direction with "
    "respect to a reference object.</li>"
    "\n\t<li>e.g. the inside of a box may be represented as Region(box, distance=INTERIOR, "
    "direction=None)</li> "
    "\n\t<li>Region(REFERENCE_OBJECT, distance=DISTANCE, direction=DIRECTION)</li>"
    "\n</ul>\n<ul>\n<p>Spatial Paths</p>"
    "\n\t<li>A SpatialPath specifies the path that some object takes during a situation. These are "
    "defined in terms of a PathOperator and whether or not the orientation changed, with respect to a "
    "reference object and an optional reference axis.</li>"
    "\n\t<li>e.g. the path of a falling object may be represented as SpatialPath(operator=TOWARD, "
    "reference_object=GROUND, reference_axis=None, orientation_changed=False)</li>"
    "\n\t<li>SpatialPath(OPERATOR, REFERENCE_OBJECT, REFERENCE_AXIS, ORIENTATION_CHANGED)</li>"
)


def main(params: Parameters) -> None:
    root_output_directory = params.creatable_directory("output_directory")
    phase1_curriculum_dir = root_output_directory / "gaila-phase-1"
    phase1_curriculum_dir.mkdir(parents=True, exist_ok=True)

    sort_by_utterance_length_flag = params.optional_boolean_with_default(
        "sort_by_utterance", default_value=False
    )
    if sort_by_utterance_length_flag:
        random_seed = params.optional_float("random_seed")
        # Using float and converting to integer until optional_integer is implemented in vistautils
        if not random_seed:
            random_seed = 1
        CurriculumToHtmlDumper().dump_to_html_as_sorted_by_utterance_length(
            GAILA_PHASE_1_CURRICULUM,
            output_directory=phase1_curriculum_dir,
            title="GAILA Phase 1 Curriculum Sorted by Utterance Length",
            random_seed=int(random_seed),
        )
    else:
        CurriculumToHtmlDumper().dump_to_html(
            GAILA_PHASE_1_CURRICULUM,
            output_directory=phase1_curriculum_dir,
            title="GAILA Phase 1 Curriculum",
        )


@attrs(frozen=True, slots=True)
class InstanceHolder:
    situation: str = attrib(validator=instance_of(str))
    """
    Holds a rendered situation string
    """
    lingustics: str = attrib(validator=instance_of(str))
    """
    Holds a rendered linguistics string
    """
    perception: str = attrib(validator=instance_of(str))
    """
    Holds a rendered perception string
    """


@attrs(frozen=True, slots=True)
class CurriculumToHtmlDumper:
    """
    Class to turn an `InstanceGroup` into an html document
    """

    def dump_to_html_as_sorted_by_utterance_length(
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
        random_seed: int,
    ):
        all_instances = []
        for (_, instance_group) in enumerate(instance_groups):
            for instance in instance_group.instances():
                (situation, dependency_tree, perception) = instance
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
                (_, speaker) = self._situation_text(situation)
                length = len(self._linguistic_text(dependency_tree, speaker).split())
                all_instances.append((situation, dependency_tree, perception, length))

        # shuffle
        random.seed(random_seed)
        random.shuffle(all_instances)
        # sort
        all_instances.sort(key=lambda instance: instance[3])

        rendered_instances = []
        for (situation, dependency_tree, perception, _) in all_instances:
            (situation_text, speaker) = self._situation_text(situation)
            rendered_instances.append(
                InstanceHolder(
                    situation=situation_text,
                    lingustics=self._linguistic_text(dependency_tree, speaker),
                    perception=self._perception_text(perception),
                )
            )

        filename = "curriculum-sorted-by-utterance.html"
        chunk_size = 50
        files_written: List[Tuple[str, str]] = []
        for i in range(0, len(rendered_instances), chunk_size):
            chunk = rendered_instances[i : i + chunk_size]
            instance_group_header = f"{int(i/chunk_size):03} - {filename}"
            relative_filename = f"{instance_group_header}.html"
            files_written.append((instance_group_header, relative_filename))
            with open(output_directory / relative_filename, "w") as html_out:
                html_out.write(f"<head>\n\t<style>{CSS}\n\t</style>\n</head>")
                html_out.write(
                    f"\n<body>\n\t<h1>{title} - Sorted by Utterance Length</h1>"
                )
                html_out.write(EXPLANATION_HEADER)
                for (instance_number, instance_holder) in enumerate(immutableset(chunk)):
                    # By using the immutable set we guaruntee iteration order and remove duplicates
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
                        f'\t\t\t\t<td valign="top">{instance_holder.situation}\n\t\t\t\t</td>\n'
                        f'\t\t\t\t<td valign="top">{instance_holder.lingustics}</td>\n'
                        f'\t\t\t\t<td valign="top">{instance_holder.perception}\n\t\t\t\t</td>\n'
                        f"\t\t\t</tr>\n\t\t</tbody>\n\t</table>"
                    )
                    html_out.write("\n</body>")

        index_file = "index-" + filename
        with open(output_directory / index_file, "w") as index_out:
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
        # first nuke the output directory
        # we check it only contains HTML files for safety
        if output_directory.exists():
            for f in output_directory.iterdir():
                if f.suffix != ".html":
                    raise RuntimeError(
                        r"Output directory does not appear to be a curriculum "
                        r"dump. It contains the non-html file {f}"
                    )
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

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
        # PreRender Instances so we can remove duplicates by converting to an immutable set
        rendered_instances = []
        for (situation, dependency_tree, perception) in instance_group.instances():
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
            (situation_text, speaker) = self._situation_text(situation)
            rendered_instances.append(
                InstanceHolder(
                    situation=situation_text,
                    lingustics=self._linguistic_text(dependency_tree, speaker),
                    perception=self._perception_text(perception),
                )
            )

        with open(output_destination, "w") as html_out:
            html_out.write(f"<head>\n\t<style>{CSS}\n\t</style>\n</head>")
            html_out.write(f"\n<body>\n\t<h1>{title} - {instance_group.name()}</h1>")
            html_out.write(EXPLANATION_HEADER)
            # By using the immutable set we guaruntee iteration order and remove duplicates
            for (instance_number, instance_holder) in enumerate(
                immutableset(rendered_instances)
            ):
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
                    f'\t\t\t\t<td valign="top">{instance_holder.situation}\n\t\t\t\t</td>\n'
                    f'\t\t\t\t<td valign="top">{instance_holder.lingustics}</td>\n'
                    f'\t\t\t\t<td valign="top">{instance_holder.perception}\n\t\t\t\t</td>\n'
                    f"\t\t\t</tr>\n\t\t</tbody>\n\t</table>"
                )
            html_out.write("\n</body>")

    def _situation_text(
        self, situation: HighLevelSemanticsSituation
    ) -> Tuple[str, Optional[SituationObject]]:
        """
        Converts a situation description into its sub-parts as a table entry
        """
        speaker = None
        output_text = [f"\n\t\t\t\t\t<h4>Objects</h4>\n\t\t\t\t\t<ul>"]
        for obj in situation.all_objects:
            property_string: str
            prop_strings = []
            if obj.properties:
                for prop in obj.properties:
                    prop_strings.append(prop.handle)
                    if prop == IS_SPEAKER:
                        speaker = obj
                property_string = "[" + ",".join(prop_strings) + "]"
            else:
                property_string = ""
            output_text.append(
                f"\t\t\t\t\t\t<li>{obj.ontology_node.handle}{property_string}</li>"
            )
        output_text.append("\t\t\t\t\t</ul>")
        if situation.actions:
            output_text.append("\t\t\t\t\t<h4>Actions</h4>\n\t\t\t\t\t<ul>")
            for acts in situation.actions:
                output_text.append(
                    f"\t\t\t\t\t\t<li>{acts.action_type.handle}</li>\n\t\t\t\t\t<ul>"
                )
                for mapping in acts.argument_roles_to_fillers.keys():
                    for object_ in acts.argument_roles_to_fillers[mapping]:
                        output_text.append(
                            f"\t\t\t\t\t\t<li>{mapping.handle} is {self._situation_object_or_region_text(object_)}</li>"
                        )
                for mapping in acts.auxiliary_variable_bindings.keys():
                    output_text.append(
                        f"\t\t\t\t\t\t<li>{mapping.debug_handle} is {self._situation_object_or_region_text(acts.auxiliary_variable_bindings[mapping])}"
                    )
            output_text.append("\t\t\t\t\t</ul>")
        if situation.always_relations:
            output_text.append("\t\t\t\t\t<h4>Relations</h4>\n\t\t\t\t\t<ul>")
            for rel in situation.always_relations:
                output_text.append(
                    f"\t\t\t\t\t\t<li>{rel.relation_type.handle}({rel.first_slot.ontology_node.handle},"
                    f"{self._situation_object_or_region_text(rel.second_slot)})</li>"
                )
            output_text.append("\t\t\t\t\t</ul>")
        return ("\n".join(output_text), speaker)

    def _situation_object_or_region_text(
        self, obj_or_region: Union[SituationObject, SituationRegion]
    ) -> str:
        if isinstance(obj_or_region, SituationObject):
            return obj_or_region.ontology_node.handle
        else:
            parts = []
            parts.append(
                f"reference_object={obj_or_region.reference_object.ontology_node.handle}"
            )
            if obj_or_region.distance:
                parts.append(f"distance={obj_or_region.distance.name}")
            if obj_or_region.direction:
                parts.append(f"direction={obj_or_region.direction}")

            return "Region(" + ", ".join(parts) + ")"

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
                        f'color: {prop.color.inverse()}; border: 1px solid black;">'
                        f"{prop.color.hex}</span>"
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
        root = ObjectPerception("root", axes=WORLD_AXES)
        graph.add_node(root)
        expressed_relations = set()

        for object_ in all_objects:
            graph.add_node(object_)
            graph.add_edge(root, object_)

        for relation_ in all_relations:
            if relation_.relation_type == PART_OF:
                graph.add_edge(relation_.second_slot, relation_.first_slot)
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
                if node.geon:
                    output_text.append(
                        f"\t\t\t\t\t\t<li>Geon: {self._render_geon(node.geon, indent_dept=7)}</li>"
                    )
                # Handle Region Relations
                for region_relation in region_relations:
                    if region_relation.first_slot == node:
                        (relation_prefix, relation_suffix) = compute_arrow(
                            region_relation, static_relations, first_frame_relations
                        )
                        relation_str = self._render_relation(
                            perception.frames[0].axis_info, region_relation
                        )
                        output_text.append(
                            f"\t\t\t\t\t\t<li>{relation_prefix}"
                            f"{relation_str}{relation_suffix}</li>"
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
                    # if matching smallerThan/biggerThan relations exist, give as single relation
                    opposite_relations = {
                        SMALLER_THAN: BIGGER_THAN,
                        BIGGER_THAN: SMALLER_THAN,
                    }
                    single_size_relation = None
                    if relation.relation_type in opposite_relations:
                        if (
                            Relation(
                                opposite_relations[relation.relation_type],
                                relation.second_slot,
                                relation.first_slot,
                            )
                            in all_relations
                        ):
                            if relation.relation_type == SMALLER_THAN:
                                single_size_relation = (
                                    f"{relation.second_slot} > {relation.first_slot}"
                                )
                            else:
                                single_size_relation = (
                                    f"{relation.first_slot} > {relation.second_slot}"
                                )
                    if single_size_relation:
                        size_output = f"\t\t\t\t\t\t<li>{relation_prefix}{single_size_relation}{relation_suffix}</li>"
                        if size_output not in output_text:
                            output_text.append(size_output)
                    else:
                        output_text.append(
                            f"\t\t\t\t\t\t<li>{relation_prefix}{relation}{relation_suffix}</li>"
                        )
            output_text.append("\t\t\t\t\t</ul>")

        if perception.during:
            output_text.append("\t\t\t\t\t<h5>During the action</h5>")
            output_text.append(self._render_during(perception.during, indent_depth=5))

        return "\n".join(output_text)

    def _render_relation(
        self, axis_info: AxesInfo[ObjectPerception], relation: Relation[ObjectPerception]
    ) -> str:
        second_slot_str: str
        filler2 = relation.second_slot
        if isinstance(filler2, Region):
            parts = [str(filler2.reference_object)]
            if filler2.distance:
                parts.append(f"distance={filler2.distance}")
            if filler2.direction:
                parts.append(
                    f"direction={sign(filler2.direction.positive)}"
                    f"{filler2.direction.relative_to_axis.to_concrete_axis(axis_info)}"
                )
            second_slot_str = f"Region({','.join(parts)})"
        else:
            second_slot_str = str(filler2)
        return f"{relation.relation_type}({relation.first_slot}, {second_slot_str})"

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

    def _linguistic_text(
        self, linguistic: LinearizedDependencyTree, speaker: Optional[SituationObject]
    ) -> str:
        """
        Parses the Linguistic Description of a Linearized Dependency Tree into a table entry

        Takes a `LinearizedDependencyTree` which is turned into a token sequence and
        phrased as a sentence for display. Returns a List[str]
        """
        if speaker:
            return (
                f'{speaker.ontology_node.handle} says: "'
                + " ".join(linguistic.as_token_sequence())
                + '"'
            )
        else:
            return " ".join(linguistic.as_token_sequence())

    def _render_geon(self, geon: Geon, *, indent_dept: int = 0) -> str:
        indent = "\t" * indent_dept
        lines = [f"{indent}<ul>"]
        lines.append(
            f"{indent}\t<li>Cross Section: {geon.cross_section} | Cross Section Size: {geon.cross_section_size}</li>"
        )
        if geon.generating_axis == geon.axes.primary_axis:
            lines.append(
                f"{indent}\t<li><b>Generating Axis: {geon.generating_axis}</b></li>"
            )
        else:
            lines.append(f"{indent}\t<li>Generating Axis: {geon.generating_axis}</li>")
        if geon.axes.orienting_axes:
            lines.append(f"{indent}\t<li>Orienting Axes:")
            lines.append(f"{indent}\t<ul>")
            for axis in geon.axes.orienting_axes:
                if axis == geon.axes.primary_axis:
                    lines.append(f"{indent}\t\t<li><b>{axis}</b></li>")
                else:
                    lines.append(f"{indent}\t\t<li>{axis}</li>")
            lines.append(f"{indent}\t</ul>")
            lines.append(f"{indent}\t</li>")
        if geon.axes.axis_relations:
            lines.append(f"{indent}\t<li>Axes Relations:")
            lines.append(f"{indent}\t<ul>")
            for axis_relation in geon.axes.axis_relations:
                if isinstance(axis_relation.second_slot, Region):
                    lines.append(
                        f"{indent}\t\t<li>{axis_relation.relation_type}({axis_relation.first_slot.debug_name},{axis_relation.second_slot})</li>"
                    )
                else:
                    lines.append(
                        f"{indent}\t\t<li>{axis_relation.relation_type}({axis_relation.first_slot.debug_name},{axis_relation.second_slot.debug_name})</li>"
                    )
            lines.append(f"{indent}\t</ul>")
            lines.append(f"{indent}\t</li>")
        lines.append(f"{indent}</ul>")
        return "\n".join(lines)


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

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _index_to_setmultidict(
    items: Iterable[_VT], index_func: Callable[[_VT], _KT]
) -> ImmutableSetMultiDict[_KT, _VT]:
    return immutablesetmultidict((index_func(x), x) for x in items)


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
