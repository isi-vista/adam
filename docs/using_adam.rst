..
   adam tutorial for open-source users

##########
Using ADAM
##########

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   using_adam_tutorials/running_experiments
   using_adam_tutorials/adding_a_curriculum
   using_adam_tutorials/adding_an_object
   using_adam_tutorials/adding_a_property
   using_adam_tutorials/adding_a_relation
   using_adam_tutorials/adding_an_action

.. autosummary::
   :toctree: _autosummary

So you want to know how to use ADAM. Here's where to start.
This page hosts a collection of tutorials on how to run experiments with ADAM,
how to define experiments,
and how to extend ADAM so it can run new kinds of experiments.
We first discuss running existing experiments,
then move on to defining new experiments and new curricula.
We then cover how to extend ADAM with new objects, properties, relations, and actions.

Note that it is also possible to extend ADAM's existing learning algorithms to learn new kinds
(for example, implementing the cross-situational learner for properties),
to add new learning algorithms (like the existing subset, pursuit, etc. algorithms) to ADAM,
or to extend ADAM by adding a new language.
However, these are out of scope for the current document.

..
   Note: Throughout these documents I have advised users to modify phase1_ontology when they want to add things.
   Is this misleading? Is there a better way to extend ADAM?
   Does it make sense to discuss caveats of this in this document?
   Do I need to discuss adding new ontologies?