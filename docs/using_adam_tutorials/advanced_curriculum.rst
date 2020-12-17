############################################
Advanced curricula: Relations and actions
############################################

We have seen already how to create simple curricula involving only objects,
however ADAM supports more complex situations as well.
You can specify spatial relationships between objects, for example,
and you can specify actions that happen in the scene.
To organize these more complex curricula,
you can also call on existing curriculum functions
or break yours up into smaller functions for convenience.

In this tutorial, we'll go over the creation of a more complex curriculum.
We'll start by showing our learner each object by itself
so that it can learn to recognize objects.
To do this we'll make use of an existing curriculum function.
Next, we'll show it various objects on tables,
then -- finally -- a person or animal putting an object on a table.
We'll assume you already understand how to make simple curricula.

******************************
An advanced example curriculum
******************************

To start our curriculum, we first need to create our curriculum function, as before.
First, we want to show the learner each object by itself.
We could do this by hand,
but that would be tedious.
Luckily, ADAM already includes a curriculum for this
called :py:func:`_make_each_object_by_itself_curriculum`,
and since curricula are built with functions,
we can just call the curriculum builder to get some instance groups::

    def build_toy_ball_train_curriculum(
        num_samples: Optional[int],
        num_noise_objects: Optional[int],
        language_generator: LanguageGenerator[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
        ],
    ) -> Sequence[Phase1InstanceGroup]:
        each_object_instance_groups = _make_each_object_by_itself_curriculum(num_samples, num_noise_objects, language_generator)

        return each_object_instance_groups

Next, let's start defining our own curriculum on top of this.
We'll define some object variables and we'll define a template for our first type of situation:
An object on a table.
We'll create this template the same way we created simpler curriculum templates.
The difference is that we're going to add a *relation* between two of our variables;
the object, and the table it's on.
A relation generally represents some spatial relationship between objects.
For example, that a truck is bigger than a watermelon,
or that a book is on a table.

ADAM provides several convenience functions for working with relations.
We're going to use just two:
`on` and `flatten_relations`.

`on` is a convenience function that will let us create the appropriate relation between our objects.
`on` is part of a family of relation DSL (domain-specific language) convenience functions;
other functions in this family include, for example, `near`, `far`, `over`, `under`, and `contact`.
These functions provide an easy way to specify relationships between objects.
Note that these functions can be used not just with a pair (like :code:`on(object, table)`)
but between many pairs of objects at once.
For example, if we wanted to specify that both the relevant object
and some other noise objects are on the table,
we could write that relation compactly as :code:`on(chain([object], noise_objects), table)`.

However, since `on` can be used this way, it has to return a *sequence of relations*,
which means we have to flatten our list of relations before using it,
replacing any nested lists with their elements.
The result looks like this::

    def build_toy_ball_train_curriculum(
        num_samples: Optional[int],
        num_noise_objects: Optional[int],
        language_generator: LanguageGenerator[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
        ],
    ) -> Sequence[Phase1InstanceGroup]:
        each_object_instance_groups = _make_each_object_by_itself_curriculum(num_samples, num_noise_objects, language_generator)

        background = make_noise_objects(num_noise_objects)
        object = standard_object("object", INANIMATE_OBJECT)
        table = standard_object("table", TABLE)
        object_on_table_template = Phase1SituationTemplate(
           "an-object-on-a-table",
            salient_object_variables=[object_on_table, table],
            background_object_variables=background,
            asserted_always_relations=flatten_relations([
                on(object, table)
            ]),
        )

        return each_object_instance_groups

Now that we have a template for an object on a table,
we're ready to add the next step of our curriculum:
A person or animal putting an object on a table.
This template will look a lot like our relation template,
except that we're going to add an *action* instead of a relation.

To do this, we're going to create an `Action`.
An `Action` consists of an ontology type
together with some information about what objects are part of the action
and how they're a part of it.
We provide this using a list of pairs (semantic role, object)
associating the semantic roles of the action
(as given in the action's `ActionDescription`)
to the object variables that fill those roles in your situation.
In our case, our putter (the person or animal putting the object on the table) will be the *agent*,
the object will be the *theme*
and the *goal* will be the place the agent is putting the object (on the table).
Note that these roles only mean anything to the language generator -- the learner won't see any of that.

To represent this goal, we'll also need to create a `Region`.
`Region`\ s are straightforward; to create one,
we only need to specify a *reference object*,
a *direction* relative to that object
and a *distance*.
In our case the reference object will be the table,
the direction will be "up" ("gravitational up," for simplicity),
and the distance will be "outside of the table, but touching it" (i.e. "exterior-but-in-contact").

Our final template will look like this::

    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    put_on_table_template = Phase1SituationTemplate(
        f"person-or-animal-puts-object-on-table",
        salient_object_variables=[agent, object, table],
        background_object_variables=background,
        actions=[
            Action(
                PUT,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, object),
                    (
                        GOAL,
                        Region(
                            table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ],
            )
        ],
        constraining_relations=[bigger_than(table, object)]
    )

We're almost done now.
We have templates for all three scenarios we wanted to show our learner.
We just need to turn those templates into instances.
Doing so is somewhat more complicated than doing so for a simple curriculum,
but not much more difficult.

To get our instances, we're going to instance all of our templates
and combine the results with the "object by itself" instances from the curriculum function we called.
We'll again use `phase1_instances`, `sampled`, and `all_possible` to get our template instances.
Finally, we'll collect our instance groups together in a list.

The result will look like this::

    def build_toy_ball_train_curriculum(
        num_samples: Optional[int],
        num_noise_objects: Optional[int],
        language_generator: LanguageGenerator[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
        ],
    ) -> Sequence[Phase1InstanceGroup]:
        each_object_instance_groups = _make_each_object_by_itself_curriculum(num_samples, num_noise_objects, language_generator)

        background = make_noise_objects(num_noise_objects)
        object = standard_object("object", INANIMATE_OBJECT)
        table = standard_object("table", TABLE)
        object_on_table_template = Phase1SituationTemplate(
           "an-object-on-a-table",
            salient_object_variables=[object_on_table, table],
            background_object_variables=background,
            asserted_always_relations=flatten_relations([
                on(object, table)
            ]),
        )

        agent = standard_object("agent", THING, required_properties=[ANIMATE])
        put_on_table_template = Phase1SituationTemplate(
            f"person-or-animal-puts-object-on-table",
            salient_object_variables=[agent, object, table],
            background_object_variables=background,
            actions=[
                Action(
                    PUT,
                    argument_roles_to_fillers=[
                        (AGENT, agent),
                        (THEME, object),
                        (
                            GOAL,
                            Region(
                                table,
                                distance=EXTERIOR_BUT_IN_CONTACT,
                                direction=GRAVITATIONAL_UP,
                            ),
                        ),
                    ],
                )
            ],
            constraining_relations=[bigger_than(table, object)]
        )

        return [
            each_object_instance_groups,
            phase1_instances(
                "objects on tables",
                sampled(
                    object_on_table_template,
                    max_to_sample=num_samples,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                ) if num_samples else all_possible(
                    object_on_table_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                language_generator=language_generator,
            ),
            phase1_instances(
                "objects put on tables",
                sampled(
                    put_on_table_template,
                    max_to_sample=num_samples,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    block_multiple_of_the_same_type=True,
                ) if num_samples else all_possible(
                    put_on_table_template,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                ),
                language_generator=language_generator,
            ),
        ]

That's it. We now have a curriculum where we show our learner each object by itself,
then objects on tables,
then a person or animal putting an object on a table,
as desired.

You've now learned all you need to know to start defining curricula
using actions, relations, and calls to create other instance groups.