#######################
Adding a new curriculum
#######################

There are a lot of possible curricula you could train a learner on,
so ADAM probably doesn't define the specific curriculum you want.
Thus, it's likely you'll want to define custom curricula for your experiments, whatever they are.
Adding curricula is relatively simple but takes some getting used to.
We will first walk through a simple example curriculum.
We'll then discuss how to create more complicated curricula.

***************************
A simple example curriculum
***************************

To illustrate how to create a new curriculum, we're going to walk you through how to create a simple curriculum.
We're going to show the learner a series of toy balls on the ground,
each described as "a ball."
We're then going to test the learner by asking it to describe a toy ball of a different color.

Curricula and curriculum functions
----------------------------------

As you might recall, a *curriculum* in ADAM consists of a series of learning examples -- pairs of *perceptual
representations* (what the learner "sees") and *linguistic descriptions* (what the learner "hears").
Every curriculum has a *training part*. The learner will be shown each training example before it is asked to describe
the perception again.
A curriculum can also include a *testing part*. The learner will be asked to describe each perception in this part after
completing the training part, but will not be shown the linguistic descriptions.

To define a new curriculum for ADAM, we have to define a *curriculum function* to generate the training part
and (if we want one) another curriculum function to generate the testing part.
We then have to register our curriculum in the experiment-running code.

A curriculum function creates a sequence of example situations paired with their perceptual representations
and linguistic descriptions.
Our curriculum functions are going to look like this:

.. code-block:: python

   def build_toy_ball_train_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ...

   def build_toy_ball_test_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ...

Situation templates
-------------------

To create our curriculum, we're going to make use of ADAM's *situation templates*.
These provide a compact way of describing similar situations,
like "a round object on a table,"
or "Mom eating a cookie (with some random unrelated objects in the scene)."
In our case, we want to describe "a ball."
When training, we want a ball that's not red,
and when testing, we want a ball that is.

Object variables
~~~~~~~~~~~~~~~~

To use situation templates, we first define some *template object variables*.
These represent objects in the situation.
These variables can be concrete ("a ball" or even "a blue ball") or abstract ("an inanimate object").
To do this we'll use the helpers :any:`standard_object` and :any:`make_noise_objects`.
:any:`standard_object` creates a single variable.
You'll be using it a lot.
:any:`make_noise_objects` on the other hand gives a sequence of variables
representing random objects.
These noise objects are useful for testing a learner's ability to learn
when there are objects in the scene unrelated to what's being described.

Our training curriculum is going to define variables like this:

.. code-block:: python

   def build_toy_ball_train_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL, banned_properties=[RED])

Note that we explicitly exclude red balls from this so that we can test the learner properly.
Our testing curriculum will then **require** a red ball:

.. code-block:: python

   def build_toy_ball_test_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL, required_properties=[RED])

Now that we have our variables, we can define a situation template that uses them.
To do this, we're going to create a :any:`Phase1SituationTemplate` object
that describes the kind of examples we want to generate.
We'll include a ball in the scene
together with some background objects that aren't balls.
We're also going to provide some syntax hints to the language generator
so that it knows not to describe the color of the balls,
since that's not what we want our learner to learn.

The result looks like this:

.. code-block:: python

   def build_toy_ball_train_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL, required_properties=[RED])
       template = Phase1SituationTemplate(
           "a-ball",
            salient_object_variables=[ball],
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BALL]),
            syntax_hints=[IGNORE_COLORS],
       )

The testing curriculum template looks similar:

.. code-block:: python

   def build_toy_ball_test_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL, banned_properties=[RED])
       template = Phase1SituationTemplate(
           "a-ball",
            salient_object_variables=[ball],
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BALL]),
            syntax_hints=[IGNORE_COLORS],
       )

Now that we have a template written, we can use it to create our examples.
More specifically, we're going to use this template to create a sequence of *instance groups*.
First, we convert the template into some specific *situations*,
which represent human-readable descriptions of specific examples.
Second, we use those situations to create *instances* --
situations
together with perceptual representations
and linguistic descriptions
that the learner can use.
To create specific situations, we're going to use the helper functions :any:`sampled` and :any:`all_possible`.
:any:`sampled` will let us randomly generate sample situations from the template.
Meanwhile, :any:`all_possible` lets us create (as the name suggests)
all possible situations that the template describes.

.. warning::

   Be careful with `all_possible`.
   For complex templates, this helper function may create many more situations than you want.
   The space of situations that can be generated from a template
   grows very quickly in the number of variables.
   Most of the time, you will want to run experiments using an explicit number of samples
   (and using `sampled`) to avoid this.

The resulting functions will look something like this:

.. code-block:: python

   def build_toy_ball_train_curriculum(
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
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BALL]),
            syntax_hints=[IGNORE_COLORS],
       )
       return [
           phase1_instances(
               "balls with some random things in the background",
               sampled(
                   template,
                   max_to_sample=num_samples,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
                   block_multiple_of_the_same_type=True,
               ) if num_samples else all_possible(
                   template,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
               ),
               language_generator=language_generator,
           ),
       ]

   def build_toy_ball_test_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL, added_properties=[RED])
       template = Phase1SituationTemplate(
           "a-ball",
            salient_object_variables=[ball],
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BALL]),
            syntax_hints=[IGNORE_COLORS],
       )
       return [
           phase1_instances(
               "balls with some random things in the background",
               sampled(
                   template,
                   max_to_sample=num_samples,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
                   block_multiple_of_the_same_type=True,
               ) if num_samples else all_possible(
                   template,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
               ),
               language_generator=language_generator,
           ),
       ]

And that's it! We've defined our curriculum functions.

Finally, we need to register our curriculum.
To do that, we're going to modify :code:`adam.experiment.log_experiment`.
The function :py:func:`curriculum_from_params` defines a mapping :py:const:`str_to_train_test_curriculum`.
Add a new entry to this mapping as follows:

.. code-block:: python

   str_to_train_test_curriculum: Mapping[
       str, Tuple[CURRICULUM_BUILDER, Optional[CURRICULUM_BUILDER]]
   ] = {
       ...
       "my-ball-curriculum": (build_toy_ball_train_curriculum, build_toy_ball_test_curriculum),
   }

The key here, :code:`"my-ball-curriculum"`, defines the name for this curriculum.
This is the name you'll use when you run your experiment.

(Note that :code:`my_test_curriculum` can be :code:`None` if you don't want to use a test curriculum.)

You can then run your curriculum by using it as the curriculum for an experiment.
If you haven't read it already, you can read :ref:`running-experiments`,
which covers how to define and run experiments.

Defining more complex curricula
-------------------------------

Curricula with more than one template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we now want to show our learner a ball, then a box. How do we do it?

For more complex curricula, like this one,
we need to define and use more than one situation template.
This works exactly the same as using a single template with one difference:
You have to convert each template into situations separately
and combine the results
before creating instances.
This is done as follows:

.. code-block:: python

   def build_ball_and_box_train_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ball = standard_object("ball", BALL)
       box = standard_object("box", BOX)
       ball_template = Phase1SituationTemplate(
           "a-ball",
            salient_object_variables=[ball],
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BALL]),
            syntax_hints=[IGNORE_COLORS],
       )
       box_template = Phase1SituationTemplate(
           "a-box",
            salient_object_variables=[box],
            background_object_variables=make_noise_objects(num_noise_objects, banned_ontology_types=[BOX]),
            syntax_hints=[IGNORE_COLORS],
       )
       return [
           phase1_instances(
               "balls with some random things in the background",
               sampled(
                   ball_template,
                   max_to_sample=num_samples,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
                   block_multiple_of_the_same_type=True,
               ) if num_samples else all_possible(
                   ball_template,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
               ),
               language_generator=language_generator,
           ),
           phase1_instances(
               "boxes with some random things in the background",
               sampled(
                   box_template,
                   max_to_sample=num_samples,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
                   block_multiple_of_the_same_type=True,
               ) if num_samples else all_possible(
                   box_template,
                   chooser=PHASE1_CHOOSER_FACTORY(),
                   ontology=GAILA_PHASE_1_ONTOLOGY,
               ),
               language_generator=language_generator,
           ),
       ]

Further notes
~~~~~~~~~~~~~

In this tutorial we focused on simple curricula with examples involving only objects.
However, ADAM supports more complex situations and examples.
For more information, please refer to the API documentation
for :any:`adam.curriculum.curriculum_utils`,
:any:`adam.situation.templates.phase1_templates`,
and :any:`adam.ontology.phase1_ontology`.

**********
Conclusion
**********

Now you should be ready to define your own curricula.
Depending on your needs, you may need to extend ADAM further.
For example, you may need new objects or properties to define your curricula.
Whatever experiments you choose run, defining curricula will remain useful.