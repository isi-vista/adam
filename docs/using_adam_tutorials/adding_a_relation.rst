#############################
Adding a new spatial relation
#############################

ADAM supports several spatial relations already,
for example "on," "under," and "over,"
however you may find that the spatial relations you need are not supported.
In general adding such relations is nontrivial.
However, adding new region-based spatial relations is relatively simple.

As an example,
we're going to go over how to add a relation "on the side of"
as a region relation in ADAM.

***********************
Creating the relation
***********************

There are two steps to defining a region relation in ADAM.
First, we'll need to define the region relation itself.
This is straightforward, as we'll see in a moment.
Second, for each language we want to use with the relation,
we'll need to change the appropriate language generator to generate language for that relation.
This is significantly more complicated than for other kinds of extension.

Defining the region relation
----------------------------

We'll start by defining the region relation.
To do this, we're going to use the helper function `make_dsl_region_relation`.
To use this function, we simply have to define how to create a region given a reference object.
We can optionally use some relation-specific keyword arguments in creating the region as well.

.. note::

   There is a similar helper function for symmetric region relations:
   `make_symmetric_dsl_region_relation`.
   It works very similarly to `make_dsl_region_relation`.
   However, since "on the side of" isn't a symmetric relation, we don't make use of it here.

We're going to represent our "on the side of" region as follows.
The direction will be along a horizontal axis of the reference object.
The specifics of this direction can be chosen using keyword arguments.
The distance will be "outside of but touching" (`EXTERIOR_BUT_IN_CONTACT`).

To choose the horizontal axis, we're going to use an :dfn:`axis function`, `HorizontalAxisOfObject`.
An axis function represents an abstract axis
that can be made concrete in reference to a specific object.
This function depends on a specific object (in our case, the reference object)
and which horizontal axis we want (there should always be exactly two).

..
  Is it true that there should always be exactly two in ADAM?

Pulling these thoughts together, we'll create our region relation function as follows::

    def _on_the_side_of_region_factory(
        reference_object: _ObjectT, *, side: int = 0, positive_direction: bool = True
    ) -> Region[_ObjectT]:
        direction = Direction(
            positive_direction,
            HorizontalAxisOfObject(reference_object, side),
        )
        return Region(
            reference_object=reference_object, distance=EXTERIOR_BUT_IN_CONTACT, direction=direction
        )


    on_the_side_of = make_dsl_region_relation(_on_the_side_of_region_factory)  # pylint:disable=invalid-name

Modifying the language generator
--------------------------------

Now that we've defined our region relation,
we need to tell ADAM to describe it.
This process is more complicated than for objects, properties, or actions,
as there is no lexicon entry system for relations.

In general, the two methods we need to modify are

1. :code:`relation_to_X_modifier`,
2. :code:`_Y_for_region_as_goal`, and

where X and Y are whatever you would use to express the relationship in the language.
The language generators use these methods to decide
(1) how to describe an `IN_REGION` relation,
and (2) how to describe a region that is the goal of an action.
Each function returns a string giving the specific word or phrase
used to express the region or relation in context.
For English, these are :code:`relation_to_prepositional_modifier` and :code:`_preposition_for_region_as_goal`.
In our case, we want them to return :code:`"on the side of"` when our new relation is used.

..
  Should we mention that X = Y = localiser for Chinese?
  Not sure it's relevant since we're not modifying the Chinese language generator.

..
  This section feels rushed,
  but at the same time it feels like it would be "too much" to describe the code changes precisely,
  since the methods involved have so many branches.
  Let me know if there's anything more you think I should describe here.

We'll change (1) first. We're going to add to the if branch that handles the "on" relationship::

        def relation_to_prepositional_modifier(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ) -> Optional[DependencyTreeToken]:
            ...
            elif region.direction:
                direction_axis = region.direction.relative_to_concrete_axis(
                    self.situation.axis_info
                )

                if region.distance == EXTERIOR_BUT_IN_CONTACT:
                    # change: add "on the side of" sub-branch
                    if isinstance(region.direction.relative_to_axis, HorizontalAxisOfObject):
                        preposition = "on the side of"
                    # was: if region.direction.positive:
                    elif region.direction.positive:
                        preposition = "on"
            ...

Next, we'll modify (2). Again, we'll put our change around the "on" case::

We'll change (1) first. We're going to add to the if branch that handles the "on" relationship::

        def _preposition_for_region_as_goal(self, region: SituationRegion) -> str:
            """
            When a `Region` appears as the filler of the semantic role `GOAL`,
            determine what preposition to use to express it in English.
            """
            if region.distance == INTERIOR:
                return "in"
            elif (
                region.distance == EXTERIOR_BUT_IN_CONTACT
                and region.direction
                and region.direction.positive
                # constrain the axis so it doesn't handle "on the side of"
                and (region.direction == GRAVITATIONAL_UP
                or region.direction == GRAVITATIONAL_AXIS_FUNCTION)
            ):
                return "on"
            # add a branch for "on the side of"
            elif (
                region.distance == EXTERIOR_BUT_IN_CONTACT
                and isinstance(region.direction, HorizontalAxisOfObject)
            ):
                return "on the side of"

**********
Conclusion
**********

We have now added "on the side of" as a relation we can use in curricula.
If we now use our new relation in a curriculum and run an experiment with it,
we should see the "canonical" descriptions use "on the side of" to describe the situation.
You should now be ready to add any region relations you should need in ADAM.
While the process is more involved than it is for extending ADAM in other ways,
I hope this tutorial has made this process clear.

.. warning::

   Because this process involves code changes to the language generators,
   it is easy to break existing descriptions.
   Be careful when adding new relations,
   and be sure to run the unit test cases (:code:`make test`)
   to make nothing is clearly broken.