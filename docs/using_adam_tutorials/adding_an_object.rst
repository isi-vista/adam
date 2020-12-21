###################
Adding a new object
###################

ADAM provides a variety of objects to use in your curricula.
However, you may find that ADAM doesn't have the right objects for your experiments.
Fortunately, adding a new kind of object to ADAM is straightforward (if potentially time-consuming).
There are only two main steps.
First, you must define the object: Its properties, physical structure, and relative size.
Second, you must tell ADAM how to describe the object in whatever languages you use for your experiment.

In this tutorial, we're going to add a toy block to show off the process.

*******************
Defining the object
*******************

The process of defining an object in ADAM is simple.
First, you need to define an *ontology type* for the object,
which describes the kind of thing that this object is.
Second, you need to define a *schema* for the object,
which describes its physical structure in detail.
Finally, you must define its size relative to the other objects in the curriculum.

Creating an ontology type
-------------------------

ADAM uses ontology types as a convenience for users to make creating curricula easier. It means that you can specify,
for example, that you want a scene that shows liquid in a cup, or an animal eating a cookie, rather than having to
specify explicit types. Note that the learner does not observe or interact with these ontology types at all.

First, we'll define the ontology type of a toy block in :any:`adam.ontology.phase1_ontology`.
To do this, we're going to use :any:`subtype` to define a subtype of :py:const:`INANIMATE_OBJECT`.
We want to give it some properties.
First, we're going to say it can fill a slot in a template,
which means that our learners can learn the object.
Second, we're going to say that it's something a person can have.

The ontology type definition then looks like this:

.. code-block:: python

   TOY_BLOCK = OntologyNode(
       "toy-block",
       [
           CAN_FILL_TEMPLATE_SLOT,
           PERSON_CAN_HAVE,
       ],
   )
   subtype(TOY_BLOCK, INANIMATE_OBJECT)

.. note::

  ADAM also has :py:const:`PERSON` and :py:const:`NONHUMAN_ANIMAL` types if you want to add objects of those kinds.

Second, we need to define our schema for toy blocks.
(This is typically done inside a function for organization purposes,
but that's not strictly necessary.)
An object schema requires, at minimum, an ontology node and a *geon*.
The geon describes the shape of the object: Its cross-sections and its *axes*.
A ball, for example, has circular cross sections that start small, get big, and end small again.

We define these using cross-sections (defined in :any:`adam.geon`) and axes (from :any:`adam.axes`).
The axes represent the object's three relative dimensions.
(One of these is always called the *primary axis* -- typically but not always the longest.
However, this distinction is mostly meaningless.)
These axis can also be related to each other.
Our toy block is going to be a thin rectangular one, so we'll include those relations when we define our block.

To define our toy block's geon,
we're going to use the :py:const:`RECTANGULAR` cross-section (from :any:`adam.geon`)
and the helper functions :any:`symmetric` and :any:`symmetric_vertical` from :any:`adam.axes`
to create our axes.
These axis helper functions take a name for the axis and return a new axis of the appropriate kind.
(Note that :any:`adam.axes` supports other kinds of axis, such as :any:`straight_up`,
and includes other helper functions for such axes.)

Our schema then looks like this:

.. code-block:: python

   def _make_toy_block_schema() -> ObjectStructuralSchema:
       front_to_back = symmetric("front-to-back")  # the "long" axis
       top_to_bottom = symmetric_vertical("top-to-bottom")
       side_to_side = symmetric("side-to-side")

       return ObjectStructuralSchema(
           ontology_node=TOY_BLOCK,
           geon=Geon(
               cross_section=RECTANGULAR,
               cross_section_size=CONSTANT,
               axes=Axes(
                   primary_axis=front_to_back,
                   orienting_axes=[top_to_bottom, side_to_side],
                   axis_relations=[
                       much_bigger_than(front_to_back, side_to_side),
                       bigger_than(side_to_side, top_to_bottom),
                   ],
               ),
           ),
       )

   _TOY_BLOCK_SCHEMA = _make_toy_block_schema()

Once we have our schema, we also have to add it to the list of structural schemata:

.. code-block:: python

   GAILA_PHASE_1_ONTOLOGY = Ontology(
       "gaila-phase-1",
       _ontology_graph,
       structural_schemata=[
           ...
           (BOOK, _BOOK_SCHEMA),
           (TOY_BLOCK, _TOY_BLOCK_SCHEMA),
           (HOUSE, _HOUSE_SCHEMA),
           ...
       ]
       ...

Finally, we must define its size relative to the existing object kinds:

.. code-block:: python

   GAILA_PHASE_1_SIZE_GRADES: Tuple[Tuple[OntologyNode, ...], ...] = (
       ...
       (BALL, BIRD, BOOK, COOKIE, CUP, HAT, JUICE, WATER, MILK, TOY_BLOCK),
       ...
   )

Be aware that depending on how you define the object's schema learners may get confused.
You may accidentally define an object that is very similar to an existing object,
which may confuse the learners.
This happened, for example, when we added a watermelon object.
The learner could not distinguish watermelons from balls.
If you are not specifically testing the learners' ability to distinguish similar things,
make sure your new object has a schema that is sufficiently distinct from other similar objects.

***************************************
Telling ADAM how to describe the object
***************************************

Once you have defined your object for ADAM, you must tell ADAM how to describe it.

To do this, you'll need to edit the *lexicon* for each language you're using.
By default, ADAM supports English and Chinese. The corresponding lexicons
are defined in :any:`adam.language_specific.english.english_phase1_lexicon`
and :any:`adam.language_specific.chinese.chinese_phase1_lexicon`, respectively.
These define mappings from ontology nodes (as defined in the previous section)
and *lexicon entries*, which tell ADAM how to describe the corresponding thing.

The English lexicon entry for our toy block will look like this:

.. code-block:: python

    LexiconEntry("block", NOUN, plural_form="blocks")

.. note::

   Lexicon entries are allowed to use more than one word.
   However, ADAM will treat these descriptions as a single word (or token).

.. note::

   ADAM also supports lexicon entries for objects representing specific, named people or things.
   For such objects we use the PROPER_NOUN tag and don't need to provide a plural:

   .. code-block:: python

       LexiconEntry("Mom", PROPER_NOUN)

We'll add it to the lexicon, :py:const:`GAILA_PHASE_1_ENGLISH_LEXICON`, between :code:`BOOK` and :code:`HOUSE`:

.. code-block:: python

   GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
       ontology=GAILA_PHASE_1_ONTOLOGY,
       ontology_node_to_word=(
           ...
           (BOOK, LexiconEntry("book", NOUN, plural_form="books")),
           (TOY_BLOCK, LexiconEntry("block", NOUN, plural_form="blocks")),
           (HOUSE, LexiconEntry("house", NOUN, plural_form="houses")),
           ...
       ),
   )

Now ADAM should support our toy block object!

**********
Conclusion
**********

In this tutorial you saw how to define a simple object.
The process remains roughly the same for objects with more complicated structure,
though some of the steps need to be repeated.
For such complex objects you must also define *subobjects* for their parts (like a human's arms).
For examples of how this is done, see :py:const:`_TABLE_SCHEMA` and :py:const:`_DOG_SCHEMA`.
Whatever object you want to add,
I hope this has made the process of doing so clearer.