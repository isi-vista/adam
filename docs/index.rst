.. adam documentation master file

####################################################
ADAM: Abduction to Demonstrate an Articulate Machine
####################################################

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. autosummary::
   :toctree: _autosummary

ADAM is a demonstration of "grounded language acquisition,"
which is to say learning (some amount of) language
from observing how language is used in concrete situations,
like infants (presumably) do.

In the next section, we present an overview of the system architecture.
That is followed by low-level API documentation.
For installation instructions and information about contributing,
please see the Markdown files in `the project's Git repository <https://github.com/isi-vista/adam/issues>`_.

ADAM is ISI's effort under DARPA's Grounded Artificial Intelligence Language Acquisition (GAILA) program.
Background for the GAILA program is given in `DARPA's call for proposals <https://www.fbo.gov/utils/view?id=b084633eb2501d60932bb77bf5ffb192>`_
and `here is a video <https://youtu.be/xGsIKHKqKdw>`_ of a talk giving an overview of our plans for ADAM
(aimed at an audience familiar with the GAILA program).

************
Introduction
************

The goal of ADAM is to build a language learning module
which can

- learn (parts of) human language
  by observing a sequence of *situations* paired with situationally-appropriate *language*
  (usually a description of the situation).
- be presented with novel situations and give reasonable linguistic descriptions of them.

For example, if the learner has been shown instances of various objects on a table
and has also been shown instances of a toy truck,
then when given the situation of a toy truck on a table,
it should be able to describe it as *truck on a table*
even if it has never seen that particular situation before.

A few aspects of ADAM's approach worth noting are:

- there are several ways one could represent a "situation."
  Rather than using, for example, videos of real-life situations,
  we represent situations by a data structure in the computer's memory.
- it is implausible for the learner to observe the data structure representing the situation directly.
  Instead, the learner will observe a *perceptual representation* derived from the situation
  and based on those perceptual primitives that there is evidence an infant has access to.
- in the GAILA program, researchers are permitted to control the type and sequence of example situations
  which are given to the learner (the *curriculum*).
  Rather than specifying a curriculum manually, we provide ways to procedurally generate curricula.

*******************
System Architecture
*******************
A particular "run" of the ADAM system is described by an `Experiment`.
Every `Experiment` needs to know

- what `LanguageLearner` to use,
- what `Situation`\ s to present to the `LanguageLearner` to teach it,
- what `Situation`\ s to present to the `LanguageLearner` to evaluate it,
- what sort of analyses to perform on the result.

There are a variety of ways to specify the situations for training and testing,
but this is prototypically done by generating them procedurally
using a `SituationTemplateProcessor`.
The way a `Situation` is presented to the `LanguageLearner` is controlled
by a `LanguageGenerator` and a `PerceptualRepresentationGenerator`.

The analyses to perform on results are given by `DescriptionObserver`\ s.

Currently the ADAM system has no entry points
and is interacted with entirely by unit tests,
but this will eventually change.
In particular, there will eventually be an interactive demonstration
which supports 3D rendering of scenes.

*****************
API Documentation
*****************

In the documentation below, because we have tried to order the modules in a recommended reading order,
sub-modules of the same package will not necessarily be adjacent.

adam.situation
---------------
.. automodule:: adam.situation

adam.language
-----------------------------
.. automodule:: adam.language

adam.perception
-----------------------------
.. automodule:: adam.perception

adam.learner
-----------------------------
.. automodule:: adam.learner

adam.situation.templates
-----------------------------
.. automodule:: adam.situation.templates

adam.language.language_generator
-------------------------------------
.. automodule:: adam.language.language_generator

adam.ontology
-----------------------------
.. automodule:: adam.ontology

adam.language.lexicon
-----------------------------
.. automodule:: adam.language.lexicon

adam.language.ontology_dictionary
-------------------------------------
.. automodule:: adam.language.ontology_dictionary

adam.experiment
-----------------------------
.. automodule:: adam.experiment

adam.experiment.instance_group
-------------------------------------
.. automodule:: adam.experiment.instance_group

adam.experiment.observer
-----------------------------
.. automodule:: adam.experiment.observer

adam.ui
-----------------------------
.. automodule:: adam.ui

adam.perception.marr
-----------------------------
.. automodule:: adam.perception.marr

adam.math_3d
-----------------------------
.. automodule:: adam.math_3d

adam.random_utils
-----------------------------
.. automodule:: adam.random_utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
