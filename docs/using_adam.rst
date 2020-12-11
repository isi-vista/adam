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
   adding_a_property
   adding_a_relation
   adding_an_action

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