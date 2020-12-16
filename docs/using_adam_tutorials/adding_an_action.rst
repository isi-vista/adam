###################
Adding a new action
###################

ADAM provides a limited variety of actions to use in your curricula.
You may need for your experiments actions not already included in ADAM.
Adding actions is not too difficult.
However, be aware that not all actions can be adequately represented in ADAM by default.
For example, we can't represent "speak" as an action,
since we can't describe in ADAM
either the mouth movements
or the speech sounds.
Furthermore, not all verbs correspond to actions in ADAM;
for example, the verb "to be" cannot reasonably be represented as an action.

With these constraints in mind,
in this tutorial we're going to add an action for carrying an object.
Specifically, the action will cover an agent carrying an already-held object from one place to another.
For example, carrying a box to a table.

****************
Adding an action
****************

The general outline for creating an action is simple.
First, the same as for objects and properties,
we describe the action to ADAM as an ontology type
together with some perceptual realization.
Second, we create lexicon entries for the action for each language we want to use with the action.
We'll cover describing the action first.

Defining an ontology node for the action is simple.
We want to define an ontology node for the carry action in :code:`adam.ontology.phase1_ontology`:

.. code-block:: python

   CARRY = OntologyNode("carry")
   subtype(CARRY, ACTION)

(Note that if our action involved a transfer of possession,
we would mark that as a property of the ontology node:
:code:`CARRY = OntologyNode("carry", [TRANSFER_OF_POSSESSION])`.
Since there is no second person involved in the carry action we want to use,
we don't need to mark it like this.)

Next, we'll create an action description that tells ADAM how the action should be perceived.
To do this, we'll need to use :code:`ActionDescriptionVariable`\ s
(which work similarly to `TemplateObjectVariable`\ s)
and an :code:`ActionDescriptionFrame`
inside an :code:`ActionDescription`.

An :code:`ActionDescriptionVariable` describes a participant in some action
and the property constraints (or lack thereof) on that participant.
:code:`ActionDescriptionVariable`\ s are also used to define region and part-of constraints.
For example, we'd use these to set up the constraint that the carrying agent has a hand
(or other manipulator; hereafter "hand"),
and the relation that the "hand" is in contact with the object while the object is being carried.

To describe the carry action, we'll need four variables:
The person or animal doing the carrying;
the thing they carry;
the body part they use to carry it;
and the place they put the thing.
We'll call these the agent, theme, manipulator, and goal:

.. code-block:: python

   _CARRY_AGENT = ActionDescriptionVariable(
       THING, properties=[ANIMATE], debug_handle="carry_agent"
   )
   _CARRY_THEME = ActionDescriptionVariable(THING, debug_handle="carry_theme")
   _CARRY_GOAL = ActionDescriptionVariable(THING, debug_handle="carry_goal")
   _CARRY_MANIPULATOR = ActionDescriptionVariable(
       THING, properties=[CAN_MANIPULATE_OBJECTS], debug_handle="carry_manipulator"
   )

Next, we'll define our :code:`ActionDescriptionFrame`.
The :code:`ActionDescriptionFrame` collects together the participants in the action and their roles.
Each participant has a *semantic* (or *thematic*) *role* in the action
which describes *how* the participant takes part in the action.
These roles are *agent*, *patient*, *theme*, and *goal*.
These roles serve to identify participants in an action description in a language-independent way.
The language generators can then use this role information in various ways
when generating descriptions of a situation involving the action.
Note that like ontology types, these roles are only for creating curricula and are not visible to learners.

(For a more complete description of these semantic roles,
see `"Summary of semantic roles and grammatical relations"`__ (Thomas Payne, 2007).

.. _semantic_roles: https://pages.uoregon.edu/tpayne/EG595/HO-Srs-and-GRs.pdf

__ semantic_roles_

..
  TODO I feel like I should explain semantic roles more/better,
  because they seems to have weird technical meanings
  and the language generators use them in various ways,
  so they don't function as arbitrary symbols for a role.
  Maybe it would help to link to a source about the meanings of the semantic roles?
  Is this a useful description at all?

In our frame, we will mark the person doing the carrying as the agent,
the thing they carry as the theme,
and the place they put it as the goal:

.. code-block:: python

   ActionDescriptionFrame({AGENT: _CARRY_AGENT, THEME: _CARRY_THEME, GOAL: _CARRY_GOAL}),

Our overall :code:`ActionDescription` will make use of both our frame and our variables.
Our :code:`ActionDescriptionFrame` describes the linguistically relevant information,
while we use the :code:`ActionDescriptionVariable`\ s as part of relations and spatial paths
to describe the physical aspects of the action:
What happens?
How do objects move during the action?
(We'll use :code:`SpatialPath`\ s to describe this.)
What relationships hold before, during and after the action?
What relationships hold over the whole course of the action for it to even make sense?
:comment:`is this a good explanation of enduring conditions?`
What properties do the participants have as a result of their role in the action? :comment:`asserted_properties`
(These are used to mark, for example, the goal as stationary.)

Putting it all together, our action description is going to look like this:

..
  TODO What is the difference between TO and TOWARD as spatial path operators?
  What is the distinction supposed to represent?

..
  NOTE: Carry is ridiculously similar to *put*.
  I copied *put* to start off my action description.
  I barely had to change anything.
  However, there is one change:
  The agent moves toward the goal.
  It might be similar to another existing verb as well.

.. code-block:: python

   _CARRY_AGENT = ActionDescriptionVariable(
       THING, properties=[ANIMATE], debug_handle="carry_agent"
   )
   _CARRY_THEME = ActionDescriptionVariable(THING, debug_handle="carry_theme")
   _CARRY_GOAL = ActionDescriptionVariable(THING, debug_handle="carry_goal")
   _CARRY_MANIPULATOR = ActionDescriptionVariable(
       THING, properties=[CAN_MANIPULATE_OBJECTS], debug_handle="carry_manipulator"
   )

   _CONTACTING_MANIPULATOR = Region(
       reference_object=_CARRY_MANIPULATOR, distance=EXTERIOR_BUT_IN_CONTACT
   )

   _CARRY_ACTION_DESCRIPTION = ActionDescription(
       frame=ActionDescriptionFrame({AGENT: _CARRY_AGENT, THEME: _CARRY_THEME, GOAL: _CARRY_GOAL}),
       during=DuringAction(
           objects_to_paths=[
              (
                   _CARRY_AGENT,
                   SpatialPath(
                       operator=TO,
                       reference_source_object=Region(_CARRY_GOAL, distance=DISTAL),
                       reference_destination_object=_CARRY_GOAL,
                   ),
              ),
              (
                   _CARRY_THEME,
                   SpatialPath(
                       operator=TO,
                       reference_source_object=_CONTACTING_MANIPULATOR,
                       reference_destination_object=_CARRY_GOAL,
                   ),
               )
           ]
       ),
       enduring_conditions=[
           Relation(SMALLER_THAN, _CARRY_THEME, _CARRY_AGENT),
       ],
       preconditions=[
           Relation(IN_REGION, _CARRY_THEME, _CONTACTING_MANIPULATOR),
           # THEME is not already located in GOAL
           Relation(IN_REGION, _CARRY_THEME, _CARRY_GOAL, negated=True),
       ],
       postconditions=[
           Relation(IN_REGION, _CARRY_THEME, _CONTACTING_MANIPULATOR, negated=True),
           Relation(IN_REGION, _CARRY_THEME, _CARRY_GOAL),
       ],
       asserted_properties=[
           (_CARRY_AGENT, VOLITIONALLY_INVOLVED),
           (_CARRY_AGENT, CAUSES_CHANGE),
           (_CARRY_AGENT, MOVES),
           (_CARRY_THEME, UNDERGOES_CHANGE),
           (_CARRY_GOAL, STATIONARY),
       ],
   )

Note that these relationships (or *relations*) work just like those in situation templates,
and we can describe them the same way:
Using the relation DSL functions (like "on", or "near")
together with :code:`itertools.chain()`.
In this case, though, we have few and simple enough relations that we can just describe them directly.

..
  TODO Should I warn users to be careful about using the variables, not the roles when describing relations, etc.?
  Not sure if that's necessary but I can see how someone might get confused.

That takes care of describing the action.
Now, as the final step, we'll add carrying to our lexicon.
We'll add a lexicon entry to :code:`GAILA_PHASE_1_ENGLISH_LEXICON` after :code:`FLY`:

.. code-block:: python

   GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
       ontology=GAILA_PHASE_1_ONTOLOGY,
       ontology_node_to_word=(
           ...
           (FLY, LexiconEntry("fly", VERB, verb_form_sg3_prs="flies")),
           (CARRY, LexiconEntry("carry", VERB, verb_form_sg3_prs="carries")),
           (RED, LexiconEntry("red", ADJECTIVE)),
           ...
       ),
   )

(Note that :code:`sg3_prs` stands for "singular third-person present (form).")

We should now be able to go create situations using this carry action.

**********
Conclusion
**********

In this tutorial you saw how to define a concrete action.
The general process should be similar whatever action you want to add,
as long as it can be represented in ADAM.

For more examples of actions and their descriptions, refer to `adam.ontology.phase1_ontology`.