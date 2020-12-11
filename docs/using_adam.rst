.. adam tutorial for open-source users

##########
Using ADAM
##########

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   running_experiments

.. autosummary::
   :toctree: _autosummary

.. TODO Rewrite this introduction

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

.. Note: Throughout this I have advised users to modify phase1_ontology when they want to add things.
   Is this misleading? Is there a better way to extend ADAM?
   Does it make sense to discuss caveats of this in this document?
   Do I need to discuss adding new ontologies?

.. TODO should these code examples include imports? I included one for the standard library since it's not part of ADAM
   itself.

Adding a new object
-------------------

Adding a new kind of object to ADAM is relatively straightforward. First, you must define the object: Its properties,
physical structure, and size. Second, you must tell ADAM how to describe the object in whatever languages you use for
your experiment.

Defining the object
~~~~~~~~~~~~~~~~~~~

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

Telling ADAM how to describe the object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once we have defined the object for ADAM, you must tell ADAM how to describe it.

To do this, you'll need to edit the *lexicon* for the language you're using.
By default, ADAM supports English and Chinese. The corresponding lexicons
are defined in `adam.language_specific.english.english_phase_lexicon`
and `adam.language_specific.chinese.chinese_phase1_lexicon`, respectively.
These define mappings from ontology nodes (as defined in the previous section)
and *lexicon entries*, which tell ADAM how to describe the corresponding thing.
A lexicon entry for a general object looks like this:

.. code-block:: python

    LexiconEntry("cow", NOUN, plural_form="cows")

For objects representing specific, named people or things, an entry looks like this:

.. code-block:: python

    LexiconEntry("Mom", PROPER_NOUN)

To add your object and its lexicon entry to one of these lexicons, you'll need to change the corresponding lexicon.
In each file there will be a variable named `GAILA_PHASE_1_$LANGUAGE_LEXICON`.
It's this variable you'll need to edit. Add a lexicon entry to the lexicon as follows:

.. code-block:: python

   GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
       ontology=GAILA_PHASE_1_ONTOLOGY,
       ontology_node_to_word=(
           (BIRD, LexiconEntry("bird", NOUN, plural_form="birds")),
           # (ontology type, lexicon entry)
           (MY_OBJECT, LexiconEntry("my-object", NOUN, plural_form="my-objects")),
           (GO, LexiconEntry("go", VERB, verb_form_sg3_prs="goes")),
       ),
   )

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

A registered curriculum consists of a training curriculum and a testing curriculum. The learner learns from the training
curriculum first. Afterwards, if a testing curriculum is specified, the learner is asked to describe the examples in the
testing curriculum without learning any of their linguistic descriptions.

The training curriculum and testing curriculum are specified using *curriculum functions*. A curriculum function has the
following type signature:

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

The curriculum function creates a sequence of example situations paired with their *perceptual representations* (what
the learner "sees") and *linguistic descriptions* (what the learner "hears"). To do this, they typically make use of
ADAM's *situation templates*, which provide a compact way of describing similar situations, like "a round object on a
table," or "Mom eating a cookie (with some random unrelated objects in the scene)."

To use situation templates, we first define some *template object variables*. To do this we typically use the helpers
`standard_object` and `make_noise_objects`. `standard_object` variables can represent both abstract things, like "a
round object," and more concrete things, like "a table" or "Mom." `make_noise_objects`, on the other hand, yields a
sequence of variables representing random unrelated objects (as in the second situation template example above).

Once we have our variables, we can define a situation template that uses them. To do this, we create a
`Phase1SituationTemplate` object which names the relevant objects in the scene (together with any background objects),
the relevant relations between them, any actions in the scene, and optionally some syntax hints the language generator
can make use of (which can, for example, tell it not to include a color in its description).

Finally, once we have a template, we can use it to create a sequence of *instance groups*. First, we convert the
template into some specific *situations*, then we create *instances* from these situations that the learner can use.
To create specific situations, we typically use `sampled` with the specified number of samples, though one can also use
`all_possible` to generate (as the name suggests) all possible (representable) situations that the template describes.
To create instances, we use the helper function `phase1_instances`. The result looks something like this:

.. code-block:: python

   def build_my_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL)
       template = Phase1SituationTemplate(
           "a-ball",
           salient_object_variables=[ball],
           background_object_variables=make_noise_objects(num_noise_objects),
           syntax_hints=[IGNORE_COLOR],
       )
       return phase1_instances(
           "a ball with some random things in the background"
           sampled(
               template,
               max_to_sample=num_samples,
               chooser=PHASE1_CHOOSER_FACTORY(),
               block_multiple_of_the_same_type=True,
           ) if num_samples else all_possible(
                single_object_template,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
           )),
           language_generator=language_generator,
       )

This is the bare minimum needed to define a curriculum function.

Finally, to register our curriculum, we must modify `adam/experiment/log_experiment.py`. The function
`curriculum_from_params` defines a mapping `str_to_train_test_curriculum`. Add a new entry to this mapping as follows:

.. code-block:: python

   str_to_train_test_curriculum: Mapping[
       str, Tuple[CURRICULUM_BUILDER, Optional[CURRICULUM_BUILDER]]
   ] = {
       ...
       "my_curriculum": (my_train_curriculum, my_test_curriculum),
   }

The string used as a key defines a name for this curriculum. You will use this name when you run your experiment.
(For more information on running your experiment, see `Running your experiment`_.

Note that `my_test_curriculum` can be `None` if you have no test curriculum.

You can then run your curriculum

Defining more complex curricula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex curricula, we may wish to define and use more than one situation template. This works exactly the same
as using a single curriculum with one difference: You must convert each template into situations separately and combine
the results. This is done as follows:

.. code-block:: python

   from itertools import chain

   ...

   def build_my_curriculum(
       ...
   ) -> Sequence[Phase1InstanceGroup]:
       ...
       template1 = ...
       template2 = ...
       return phase1_instances(
           "two templates"
           chain(  # use chain to combine the situations generated from each template
               sampled(
                   template1,
                   ...
               ) ...,
               sampled(
                   template2,
                   ...
               ) ...,
           )
           language_generator=language_generator,
       )


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
