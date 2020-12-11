###################
Adding a new object
###################

You may find that ADAM doesn't have the objects that you need to run your experiments.
Fortunately, adding a new kind of object to ADAM is relatively straightforward.
There are only two main steps.
First, you must define the object: Its properties, physical structure, and relative size.
Second, you must tell ADAM how to describe the object in whatever languages you use for your experiment.

.. TODO: Rewrite "Adding an object" to match new structure
.. TODO: Rewrite "Adding an object" to be more casual
.. TODO: Rewrite "Adding an object" to walk through an example

*******************
Defining the object
*******************

Defining an ontology type is simple.
In phase1_ontology, you will define a subtype of :code:`PERSON`, :code:`NONHUMAN_ANIMAL`, or
:code:`INANIMATE_OBJECT` as follows:

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
very similar to an existing object, which may confuse the learners. This happened, for example, when :code:`watermelon` was
added to the ontology: The learner could not distinguish watermelons from balls. If you are not specifically testing
the learners' ability to distinguish similar things, make sure your new object has a schema that is sufficiently
distinct from other similar objects.

***************************************
Telling ADAM how to describe the object
***************************************

Once we have defined the object for ADAM, you must tell ADAM how to describe it.

To do this, you'll need to edit the *lexicon* for the language you're using.
By default, ADAM supports English and Chinese. The corresponding lexicons
are defined in :code:`adam.language_specific.english.english_phase_lexicon`
and :code:`adam.language_specific.chinese.chinese_phase1_lexicon`, respectively.
These define mappings from ontology nodes (as defined in the previous section)
and *lexicon entries*, which tell ADAM how to describe the corresponding thing.
A lexicon entry for a general object looks like this:

.. code-block:: python

    LexiconEntry("cow", NOUN, plural_form="cows")

For objects representing specific, named people or things, an entry looks like this:

.. code-block:: python

    LexiconEntry("Mom", PROPER_NOUN)

To add your object and its lexicon entry to one of these lexicons, you'll need to change the corresponding lexicon.
In each file there will be a variable named :code:`GAILA_PHASE_1_$LANGUAGE_LEXICON`.
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
