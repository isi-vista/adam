#####################
Adding a new property
#####################

ADAM provides some properties and colors to use in your curricula.
However, you may need additional properties for your experiments.
Adding new properties is straightfoward.
There are only two main steps.
First, you must define an ontology type for the property.
Second, you must tell ADAM how to describe the property in whatever languages you use for your experiment.

In this tutorial, we're going to add a rubber property and a cyan color to show off the processes.
We might use this rubber property, for example, on the existing :py:const:`BALL` object.

************************
Adding a binary property
************************

ADAM uses ontology types as a tool for creating curricula.
Note that the learner does not observe or interact with these ontology types at all.

To define our rubber property's ontology type, we simply create an ontology node with the :py:const:`BINARY` property
that subtypes :py:const:`PERCEIVABLE_PROPERTY`:

.. code-block:: python

   RUBBER = OntologyNode("rubber", [BINARY])
   subtype(RUBBER, PERCEIVABLE_PROPERTY)

.. note::

   ADAM also supports properties that learners can't perceive.
   For example, :py:const:`EDIBLE` is such a property.
   Such properties simply subtype :py:const:`PROPERTY` directly.

Next, we must tell ADAM how to describe the property.
To do this, we'll need to edit the *lexicon* for the language(s) we're using.
By default, ADAM supports English and Chinese. The corresponding lexicons
are defined in :any:`adam.language_specific.english.english_phase1_lexicon`
and :any:`adam.language_specific.chinese.chinese_phase1_lexicon`, respectively.
These define mappings from ontology nodes (as defined in the previous section)
to *lexicon entries*, which tell ADAM how to describe the corresponding thing.

Our English lexicon entry for rubber will look like this:

.. code-block:: python

    LexiconEntry("rubber", ADJECTIVE)

We'll add it to the lexicon, :py:const:`GAILA_PHASE_1_ENGLISH_LEXICON`, between :code:`FLY` and :code:`RED`:

.. code-block:: python

   GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
       ontology=GAILA_PHASE_1_ONTOLOGY,
       ontology_node_to_word=(
           ...
           (FLY, LexiconEntry("fly", VERB, verb_form_sg3_prs="flies")),
           (RUBBER, LexiconEntry("rubber", ADJECTIVE)),
           (RED, LexiconEntry("red", ADJECTIVE)),
           ...
       ),
   )

Finally, because we want balls to be made of rubber,
we'll open up :any:`adam.ontology.phase1_ontology`
and add the :code:`RUBBER` property to :code:`BALL`:

.. code-block:: python

   BALL = OntologyNode(
       "ball",
       [CAN_FILL_TEMPLATE_SLOT, PERSON_CAN_HAVE, RUBBER, ROLLABLE, RED, BLUE, GREEN, BLACK, WHITE],
   )
   subtype(BALL, INANIMATE_OBJECT)

That's it! Now we have a rubber property and we've applied it to balls.

**************
Adding a color
**************

Adding colors is nearly as easy as adding binary properties.
There's only one additional step: Defining the color's RGB values.

We start off similarly with defining an ontology type, in this case for cyan:

.. code-block:: python

   CYAN = OntologyNode("cyan", [CAN_FILL_TEMPLATE_SLOT])
   subtype(CYAN, COLOR)

Next, we need to add an entry to :code:`COLORS_TO_RGBS`:

.. code-block:: python

   ...
   _CYAN_HEX = [(0, 255, 255)]  # (0, 254, 255)
   COLORS_TO_RGBS: ImmutableDict[
       OntologyNode, Optional[Sequence[Tuple[int, int, int]]]
   ] = immutabledict(
       [
           ...
           (DARK_BROWN, _DARK_BROWN_HEX),
           (CYAN, _CYAN_HEX),
       ]
   )

Finally, we'll add a lexicon entry the same way we did for our rubber property.
We'll add our entry to :py:const:`GAILA_PHASE_1_ENGLISH_LEXICON` after :code:`DARK_BROWN`:

.. code-block:: python

   GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
       ontology=GAILA_PHASE_1_ONTOLOGY,
       ontology_node_to_word=(
           ...
           (DARK_BROWN, LexiconEntry("dark brown", ADJECTIVE)),
           (CYAN, LexiconEntry("cyan", ADJECTIVE)),
       ),
   )

That's it. You now have a cyan color property you can use in your experiments.

**********
Conclusion
**********

Defining properties, whether binary or color, is quite simple.
I hope this tutorial has made the process clear.
