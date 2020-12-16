.. _running-experiments:

###################
Running experiments
###################

The whole point of ADAM is to run experiments. This document will show how to do that. First, we'll show how to run one
of the existing experiments. Second, we'll discuss creating new experiments, first with a high level description and
then a concrete example.

******************************
Running an existing experiment
******************************

We're going to run a simple experiment. First, make sure you've defined a :file:`root.params` file as described in the README.
(We'll wait here for you to do that.)

Next, from the repository root, do:

.. code-block:: shell

   python adam/experiment/log_experiment.py parameters/experiment/m9/attributes.params

The experiment should run, creating a directory :file:`%adam_experiment_root%/m9` holding the results of the experiment
(where :code:`%adam_experiment_root%` is whatever you set the variable to in your :file:`root.params`).

.. note::

   Whenever we use percentage signs around a variable name, like :code:`%experiment%`, that means "replace
   :code:`%experiment%` with the value of the variable :code:`experiment` in context."
   When processing a parameters file, ADAM looks up the values of these variables
   in the other parameters files listed under :code:`_includes`.
   If it can't find it there, it will try looking it up in your environment variables.

Congratulations; you've run your first experiment.

************************
Defining new experiments
************************

So now you want to define your own experiments to run.

To do this, you'll need to write *parameters files* that describe your experiments.
These parameters files tell ADAM, at a minimum,

1. the experiment name,
2. the curriculum to use for the experiment (which define the examples the learner sees during learning and the examples
   it is judged on during testing),
3. the learner (together with any of the learner's required parameters), and
4. a directory where it can save the results.

Experiments are generally run using :code:`adam.experiment.log_experiment`, passing the parameter file as an argument. This
works like we saw above.

To get you started, we've included a template parameters file, :file:`parameters/experiment/experiment_template.params`.
To start defining your experiment, simply copy the template and replace the experiment and curriculum names.

The experiment we just ran is fine, but it's using an old version of the learner code.
What if we want to use the new version?
Let's define an experiment that runs a similar experiment, but using the new-style learners.
We're going to create a parameters file :file:`parameters/experiment/m9_attributes_with_new_learner.params` that looks like this:

.. code-block:: yaml

   _includes:
      - "../root.params"

   # Meta parameters: Where to store the experiment results, and what results to collect
   experiment: "m9-attributes-with-new-learner"
   experiment_group_dir: '%adam_experiment_root%/%experiment%'

   # Some potentially useful (but optional) parameters
   # that control *how much* and *what kind of* output the experiment will produce
   hypothesis_log_dir: "%experiment_group_dir%/hypotheses"
   include_image_links: true

   sort_learner_descriptions_by_length: True
   num_pretty_descriptions: 5

   # The curriculum to use in the experiment
   curriculum: "m9-attributes"

   # The language to use in the experiment
   language_mode: ENGLISH

   # The learner setup to use in the experiment
   # As a default, we include learners for every role, using subset where available
   learner: "integrated-learner-params"
   object_learner:
      learner_type: "recognizer"
   attibute_learner:
      learner_type: "subset"
   relation_learner:
      learner_type: "none"
   action_learner:
      learner_type: "none"
   include_functional: False
   include_generics: False

We can then run this like the first experiment:

.. code-block:: shell

   python adam/experiment/log_experiment.py parameters/experiment/m9_attributes_with_new_learner.params

This should produce similar (but not quite the same!) results, again in a directory under your :code:`adam_experiment_root`.

Now you're ready to define your own experiments. Depending on what experiments you want to run, you may need to extend
ADAM before you can run them. However, this core process -- defining experiments using parameters files, then running
a script that uses those parameters -- will stay the same.

Further notes
-------------

By convention, experiment parameters files live in :file:`parameters/experiment` and its subdirectories,
but you can put them anywhere you want.

..
  Refer to Jacob's excellent documentation. Accept no substitutes.

:file:`log_experiment.py` supports many parameters; for a full description of what's available, see
:file:`adam/experiment/README.md`.