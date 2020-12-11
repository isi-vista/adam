.. adam tutorial for open-source users

##########
Using ADAM
##########

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   running_experiments
   adding_a_curriculum
   adding_an_object

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
