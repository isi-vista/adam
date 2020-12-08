.. adam tutorial for open-source users

##########
Using ADAM
##########

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. autosummary::
   :toctree: _autosummary

The following is a guide on how to use ADAM to run your own experiments. We discuss how to extend the existing framework
with new objects, properties, relations, actions, and curricula. We first give a high-level overview of how to do each
thing, then discuss a concrete example experiment which illustrates each kind of extension.

.. is this how you write a comment?
   note that TODO we actually have to include references to learners and languages as well

It is also possible to extend ADAM's existing learning algorithms to learn new kinds (for example, implementing the
cross-situational learner for properties), to add new learning algorithms (like the existing subset, pursuit, etc.
algorithms) to ADAM, or to extend ADAM by adding a new language. However, these are out of scope for the current
document.

*******************
High-level overview
*******************

Adding a new object
-------------------

Adding a new kind of object to ADAM is relatively straightforward. You must define an ontology type for the object,
define a schema for the object describing its structure, and define its size relative to existing objects.
You must also define for each language you want to use in your experiment how that language describes the object. (For
example, defining that a ball is called "ball" in English.)

Defining an ontology type is simple. In phase1_ontology, you will define a subtype of `PERSON`, `NONHUMAN_ANIMAL`, or
`INANIMATE_OBJECT` as follows:

.. code-block:: python

   MY_OBJECT = OntologyNode(
       "my-object",
       [CAN_FILL_TEMPLATE_SLOT, SOME_PROPERTY, ANOTHER_PROPERTY, ...],
   )
   subtype(MY_OBJECT, SUPERTYPE)

You then define a schema for the object:

.. code-block:: python

   def _make_my_object_schema() -> ObjectStructuralSchema:
       generating_axis = symmetric_vertical("my-object-generating")
       orienting_axis_0 = symmetric("my-object-orienting-0")
       orienting_axis_1 = symmetric("my-object-orienting-1")

       return ObjectStructuralSchema(
           ontology_node=MY_OBJECT,
           geon=Geon(
               cross_section=CIRCULAR,
               cross_section_size=CONSTANT,
               axes=Axes(
                   primary_axis=generating_axis,
                   orienting_axes=[orienting_axis_0, orienting_axis_1],
               ),
           ),
       )

   _MY_OBJECT_SCHEMA = _make_my_object_schema()

This schema must then be added to the list of structural schemata:

.. code-block:: python

   GAILA_PHASE_1_ONTOLOGY = Ontology(
       "gaila-phase-1",
       _ontology_graph,
       structural_schemata=[
           ...
           (MY_OBJECT, _MY_OBJECT_SCHEMA)
           ...
       ]
       ...

Finally, we must define its size relative to the existing object kinds:

.. code-block:: python

   GAILA_PHASE_1_SIZE_GRADES: Tuple[Tuple[OntologyNode, ...], ...] = (
       ...
       (WATERMELON, PAPER, HAND, HEAD, _ARM_SEGMENT, _LEG_SEGMENT, _FOOT),
       (MY_OBJECT,),
       (BALL, BIRD, BOOK, COOKIE, CUP, HAT, JUICE, WATER, MILK),
       ...
   )

It is important to be careful with how you define the object's schema. You may accidentally define an object that is
very similar to an existing object, which may confuse the learners. This happened, for example, when `watermelon` was
added to the ontology: The learner could not distinguish watermelons from balls. If you are not specifically testing
the learners' ability to distinguish similar things, make sure your new object has a schema that is sufficiently
distinct from other similar objects.

.. Here's what I think the general outline is:
   1. In phase1_ontology:
       1. Add a new ontology type, subtyping one of PERSON, NONHUMAN_ANIMAL, or INANIMATE_OBJECT
           1. The ontology type should have the property CAN_FILL_TEMPLATE_SLOT as well as any others you need.
       2. Add a schema
       3. Add an entry to GAILA_PHASE_1_SIZE_GRADES
   2. Add a lexicon entry to $LANGUAGE_phase1_lexicon for each langauge you want to use in your experiment

Adding a new property
---------------------

TODO

.. Here's what I think the general outline is.
   Note that this only covers non-color properties. Color properties are their own section.
   1. In phase1_ontology:
       1. Add a new ontology type subtyping PERCEIVABLE_PROPERTY.
   2. Add a lexicon entry to $LANGUAGE_phase1_lexicon for each langauge you want to use in your experiment

Adding a new color
~~~~~~~~~~~~~~~~~~

TODO

.. Here's what I think the general outline is.
   Adding a color is much like adding a property with a few differences.
   1. In phase1_ontology:
       1. Add a new ontology type subtyping COLOR, with the added property CAN_FILL_TEMPLATE_SLOT.
       2. Add an entry to COLORS_TO_RGBS mapping your color to the appropriate hex value or values.
   2. Add a lexicon entry to $LANGUAGE_phase1_lexicon for each langauge you want to use in your experiment

Adding a new relation
---------------------

TODO

.. This one's complicated. It seems nontrivial to address this generally

Adding a new action
-------------------

TODO

.. Here's what I think the general outline is:
   1. In phase1_ontology:
       1. Add a new ontology type subtyping ACTION.
           1. If it involves a transfer of possession, mark it as such.
       2. Create an action description for each variation of the action.
   2. Add a lexicon entry to $LANGUAGE_phase1_lexicon for each langauge you want to use in your experiment

Adding a new curriculum
-----------------------

Defining a new curriculum for ADAM means defining functions to generate the curriculum and registering them in the code.

A curriculum function has the following type signature:

.. code-block:: python

   def build_my_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ...

The curriculum function creates a sequence of example situations paired with their perceptual representations (what the
learner "sees") and linguistic descriptions (what the learner "hears"). To do this, they typically make use of ADAM's
*situation templates*, which provide a compact way of describing similar situations, like "a round object on a table,"
or "Mom eating a cookie (with some random unrelated objects in the scene)."

To use situation templates, we first define some template object variables. To do this we typically use the helpers
`standard_object` and `make_noise_objects`. `standard_object` variables can represent both abstract things, like "a
round object," and more concrete things, like "a table" or "Mom." `make_noise_objects`, on the other hand, yields a
sequence of variables representing random unrelated objects (as in the second situation template example above).

Once we have our variables, we can define a situation template that uses them. To do this, we create a
`Phase1SituationTemplate` object which names the objects in the scene, the relevant relations between them, any actions
in the scene, and optionally some syntax hints the language generator can make use of.

Finally, once we have a template, we can use it to create a sequence of instance groups. First, we convert the template
into some specific situations, then we create instances from these situations that the learner can use.
To create specific situations, we typically use `sampled` with the specified number of samples, though one can also use
`all_possible` to generate (as the name suggests) all possible (representable) situations that the template describes.
To create instances, we use the helper function `phase1_instances`.

For more complex curricula, we may define and use more than one situation template.

.. TODO explain how to use more than one situation template. example: phase1_curriculum.py:199.

.. Here's what I think the general outline is:
   1. Define a curriculum function for training and (optionally) testing; these functions work as follows
       1. Take num_samples: Optional[int], num_noise_objects: Optional[int],
          language_generator: LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]
          and returns a Sequence[Phase1InstanceGroup].
       2. Create any necessary template variables
       3. Use those variables to create a Phase1SituationTemplate
       4. Return phase1_instances("$curriculum_name", chain(*[sampled(template1, ...), sampled(template2, ...), ..., sampled(templatek, ...)]), language_generator=language_generator)
   2. Add a curriculum entry to str_to_train_test_curriculum in curriculum_from_params() in log_experiment.py

Running your experiment
-----------------------

Once we have defined a curriculum for an experiment, it is relatively easy to define an experiment using that
curriculum. We simply write a parameters file (a YAML file with some enhancements) describing the experiment to be run
and feed this to our experiment runner script.

.. TODO see: template

Experiment parameters files live in `parameters/experiment` and subdirectories thereof. This is for organization and is
not strictly necessary.

.. Refer to Jacob's excellent documentation. Accept no substitutes.

For a more complete description of the all the options one can configure, consult `adam/experiment/README.md`.

Note that before running any experiments, it is important to make sure that you defined a `root.params` file as
described in the README.

.. TODO walk through this root.params step in the example?

.. Here's what I think the general outline is.
   0. Make sure you have set up a root.params file as described in the README.
   1. Define a parameters file for the experiment under parameters/experiment,
      say parameters/experiment/mine/snake_toy.params.
   2. Run $ python adam/experiment/log_experiment.py parameters/experiment/mine/snake_toy.params.
   I think that this actually shows we probably want to include a template experiment parameters file.

******************
Example experiment
******************

Introduction
------------

TODO

Adding a snake toy
------------------

TODO

Making it flexible (property)
-----------------------------

TODO

Wrapping it around a truck (relation)
-------------------------------------

TODO

Wrapping it around a truck (relation)
-------------------------------------

TODO

Picking up the snake toy (action)
---------------------------------

TODO

Running the experiment
----------------------

TODO
