#######################
Adding a new curriculum
#######################

Defining a new curriculum for ADAM means defining functions to generate the curriculum and registering them in the code.

A registered curriculum consists of a training curriculum and a testing curriculum. The learner learns from the training
curriculum first. Afterwards, if a testing curriculum is specified, the learner is asked to describe the examples in the
testing curriculum without learning any of their linguistic descriptions.

The training curriculum and testing curriculum are specified using *curriculum functions*. A curriculum function has the
following type signature:

.. code-block:: python

   def build_my_curriculum(
       num_samples: Optional[int],
       num_noise_objects: Optional[int],
       language_generator: LanguageGenerator[
           HighLevelSemanticsSituation,
           LinearizedDependencyTree,
       ],
   ) -> Sequence[Phase1InstanceGroup]:
       ...

The curriculum function creates a sequence of example situations paired with their *perceptual representations* (what
the learner "sees") and *linguistic descriptions* (what the learner "hears"). To do this, they typically make use of
ADAM's *situation templates*, which provide a compact way of describing similar situations, like "a round object on a
table," or "Mom eating a cookie (with some random unrelated objects in the scene)."

To use situation templates, we first define some *template object variables*. To do this we typically use the helpers
`standard_object` and `make_noise_objects`. `standard_object` variables can represent both abstract things, like "a
round object," and more concrete things, like "a table" or "Mom." `make_noise_objects`, on the other hand, yields a
sequence of variables representing random unrelated objects (as in the second situation template example above).

Once we have our variables, we can define a situation template that uses them. To do this, we create a
`Phase1SituationTemplate` object which names the relevant objects in the scene (together with any background objects),
the relevant relations between them, any actions in the scene, and optionally some syntax hints the language generator
can make use of (which can, for example, tell it not to include a color in its description).

Finally, once we have a template, we can use it to create a sequence of *instance groups*. First, we convert the
template into some specific *situations*, then we create *instances* from these situations that the learner can use.
To create specific situations, we typically use `sampled` with the specified number of samples, though one can also use
`all_possible` to generate (as the name suggests) all possible (representable) situations that the template describes.
To create instances, we use the helper function `phase1_instances`. The result looks something like this:

.. code-block:: python

   def build_my_curriculum(
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
           background_object_variables=make_noise_objects(num_noise_objects),
           syntax_hints=[IGNORE_COLOR],
       )
       return phase1_instances(
           "a ball with some random things in the background"
           sampled(
               template,
               max_to_sample=num_samples,
               chooser=PHASE1_CHOOSER_FACTORY(),
               block_multiple_of_the_same_type=True,
           ) if num_samples else all_possible(
                single_object_template,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
           )),
           language_generator=language_generator,
       )

This is the bare minimum needed to define a curriculum function.

Finally, to register our curriculum, we must modify `adam/experiment/log_experiment.py`. The function
`curriculum_from_params` defines a mapping `str_to_train_test_curriculum`. Add a new entry to this mapping as follows:

.. code-block:: python

   str_to_train_test_curriculum: Mapping[
       str, Tuple[CURRICULUM_BUILDER, Optional[CURRICULUM_BUILDER]]
   ] = {
       ...
       "my_curriculum": (my_train_curriculum, my_test_curriculum),
   }

The string used as a key defines a name for this curriculum. You will use this name when you run your experiment.
(For more information on running your experiment, see `Running your experiment`_.

Note that `my_test_curriculum` can be `None` if you have no test curriculum.

You can then run your curriculum

*******************************
Defining more complex curricula
*******************************

For more complex curricula, we may wish to define and use more than one situation template. This works exactly the same
as using a single curriculum with one difference: You must convert each template into situations separately and combine
the results. This is done as follows:

.. code-block:: python

   from itertools import chain

   ...

   def build_my_curriculum(
       ...
   ) -> Sequence[Phase1InstanceGroup]:
       ...
       template1 = ...
       template2 = ...
       return phase1_instances(
           "two templates"
           chain(  # use chain to combine the situations generated from each template
               sampled(
                   template1,
                   ...
               ) ...,
               sampled(
                   template2,
                   ...
               ) ...,
           )
           language_generator=language_generator,
       )