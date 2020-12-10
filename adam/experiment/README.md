# Experimentation Setup
Our experiments can be influenced by a number of parameters. This guide is intended to list all potential parameters, their impacts on the relevant portion of the curriculum, and any conflicting settings. Parameters are written using the following line format:
`- param_name: type (Optional: Default) - Description`

## State Restoration
As described below, our experiments have a parameter `resume_from_latest_logged_state` which, if true, does attempt to restart an experiment from the latest logged state. However, currently the experiment may restore the observers and the learners to different states IF the 'latest state' doesn't match between these logged folders.

### Curriculum Pre-Generation
- curriculum_repository_path: Path (Optional) - A directory where pre-generated curriculum have been stored, used to save on recreating the curriculum each time an experiment runs.
    - Note that to use this parameter, the curriculum in use must be pre-generated using the script `adam.experiment.generate_curriculum`. All curriculum-related parameters must be the same when pre-generating the curriculum.

## Generic Parameter Control
These parameters are independent of any specific experiment.
### Curriculum Params
- curriculum: String - The name of the curriculum to generate
    - Valid options: "m6-deniz", "each-object-by-itself", "pursuit", "m6-preposition", "m9-objects", "m9-attributes", "m9-relations", "m9-events", "m9-debug", "m9-complete", "m13-imprecise-size", "m13-imprecise-temporal", "m13-subtle-verb-distinction", "m13-object-restrictions", "m13-functionally-defined-objects", "m13-generics", "m13-complete", "m13-verbs-with-dynamic-prepositions", "m13-shuffled", "m13-relations", "m15-object-noise-experiments"
- pursuit-curriculum-params: Namespace - The configuration for the pursuit specific curriculum
- use-path-instead-of-goal: Boolean (Optional) - If true some curriculums use a path indication rather than a goal
- num_samples: Integer (Optional: 5) - The number of each specific situation type to instantiate
- num_noise_objects: Integer (Optional: 0) - The number of noise objects in a situation

### Experiment Params
- experiment: String - The experiment name
- debug_log_directory: Path (Optional) - The path to log associated debug information, in this case it enables debug graph logging
- language_mode: String (Optional: "English") - The language the experiment should use. This string value converts to an enumerated value
    - Valid options: "English", "Chinese"
- experiment_group_dir: Path - The path where the experiment will be logged to
- learner: String - The name of the learner type to use
    - Valid options: "pursuit", "object-subset", "preposition-subset", "attribute-subset", "verb-subset", "integrated-learner", "integrated-learner-recognizer", "pursuit-gaze", "integrated-object-only"
- beam_size: Integer (Optional: 10)- Used to constrain the search space for learner algorithms

- pre_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's output of a situation before the learn process is run on the new input.
- post_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's output of a situation after the learning process is ran on the new input.
- test_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's performance on the test situations.
- hypothesis_log_dir: Path (Optional) - This path is a directory to log internal learner hypotheses into
- log_hypothesis_every_n_steps: Integer (Optional: 250)- The number of situations to observe before the system logs the internal state of the experiment
- log_learner_state: Boolean (Optional: True)- If true, save the internal trained learner state in a pickle.
- starting_point: Integer (Optional) - Useful when debugging to start training from a specific situation number in the curriculum. Please enter the LAST observed situation and the system will restart from the next situation. Can not be used if `resume_from_latest_logged_state` is true.
- point_to_log: Integer (Optional) - Useful debugging tool causes a specific log of hypotheses, learner, & observer states (future functionality for Observers)
- learner_state_path: Path (Optional) - Provide a path to a pickled learner file to load to continue an experiment with. Used commonly in combination with `starting_point`. Can not be used if `resume_from_latest_logged_state` is true.
- observers_state_path: Path (Optional) - Provide a path to a pickled observers state to load to continue an experiment with observation. Used commonly in combinationed with `starting_point`. Can not be used if `resume_from_latest_logged_state` is true.
- resume_from_latest_logged_state: Boolean (Optional: False) - Restart an experiment from the most recent logged state that can be located in the directory structure pointed to by `hypothesis_log_dir`.
- debug_learner_pickling: Boolean (Optional: False) - If true, log additional debug information for the learner when pickling

### Observer Params
Currently the accessible observers by these parameters are HTML Observers which can include other sub-observer types
- include_acc_observer: Boolean (Optional) - Include an accuracy observer
- accuracy_to_txt: Boolean (Optional) - Log accuracy information to a file
- accuracy_logging_path: String (Optional) - File path as string to log accuracy information to if `accuracy_to_txt` is true
- include_pr_observer: Boolean (Optional) - Include a Precision and Recall Observer
- log_pr: Boolean (Optional) - Log precision and recall information to a file
- pr_log_path: String (Optional) - File path as string to log precision and recall information to if `log_pr` is true

### Optional: Pegasus Workflow Params
This section of parameters is specific to using the Pegasus workflow management tool in combination with the ISI Saga cluster.
See [Pegasus Wrapper Overview][https://github.com/isi-vista/vista-pegasus-wrapper/blob/master/docs/api_overview.rst] for more details on how to use this section

## Learner Parameters
### How To Define an Integrated Learner
The integrated learner can be initialized many different ways in an experiment. The best way is to use `learner_type: "integrated-learner-params"` because this functionality extends an ability to fine tune all parameters and learner types. To use this learner type the namespaces `object_learner`, `attribute_learner`, `plural_learner`, `relation_learner`, and `action_learner` all follow the same layout below:
- learner_type: str (Optional: "subset") - Indicates which learner model will be loaded. Normally the valid options are: "subset", "pursuit", and "none"
- ontology: str (Optional: "phase2") - Indicates which ontology should be loaded with the learner

Any additional parameters for a learner type as described below should also be placed directly in the namespace. This configuration allows for detailed configuration of individual components.

Finally we have two types of of learners which are special implementations which don't directly follow the subset or pursuit models. These are functional objects learner and a generics model. To add these learners into the integrated model configure the following parameters:
- include_functional_learner: Boolean (Optional: True) - Include the functional learner model
- include_generics_learner: Boolean (Optional: True) - Include the functional learner model

As an example here's how one would specify the Propose-But-Verify Object Learner with random seed 42.
```
object_learner:
    learner_type: pbv
    random_seed: 42
```

### Subset
*Valid For: Object, Attribute, Plural, Relation, Action*
None

### Propose But Verify
*Valid For: Object*
- graph_match_confirmation_threshold: Float (Optional: 0.8) - A value between 0 and 1 which indicates how much a new hypothesis must match a previous pattern to count as a successful match.
- random_seed: Integer (Optional: 0) - A seed for the RandomChooser which will make the random decisions between possible meanings

### Cross-Situational
*Valid For: Object*
- graph_match_confirmation_threshold: Float (Optional: 0.8) - A value between 0 and 1 which indicates how much a new hypothesis must match a previous pattern to count as a successful match.
- lexicon_entry_threshold: Float - A float between 0 and 1 controlling when the learner will consider a word's meaning "known" and enter that word into its lexicon
- smoothing_parameter: Float - A small positive number added to each hypothesis score when updating hypotheses. This should be at most 0.1 and possibly much less.
- expected_number_of_meanings: Float - A positive number giving an upper bound for the number of possible abstract meanings (cat, dog, etc. rather than a cat in a scene). For the phase 1 ontology this should be set to at least 67.

### Pursuit
*Valid For: Object, Attribute, Relation*
- learning_factor: Float - The learning factor rate
- graph_match_confirmation_threshold: Float - A value between 0 and 1 which indicates how much a new hypothesis must match a previous pattern to count as a successful match.
- lexicon_entry_threshold: Float - A value between 0 and 1 which indicates what probability a concept should be allowed into the 'known' lexicon
- smoothing_parameter: Float - A value between 0 and 1 which is generally small which smooths the probabilities over time.
- random_seed: Integer (Optional: 0) - A seed for the RandomChooser which is used in this learner

## Specific Experiment Details
Any experiment specific parameters that are built into the program are explained below.

### Gaze Ablation
TODO: https://github.com/isi-vista/adam/issues/982

### Object Language Ablation
- object_learner_type: String - Defines which specific learning algorithm to implement for Objects
    - Valid options: "subset", "pbv", "cross-situational", "pursuit"
- learner_params: Namespace - Defines a set of parameters to configure the different learner types
- train_curriculum: Namespace - Defines a set of parameters to configure the training curriculum for this experiment
    - accurate_language_percentage: Float - A value between 0 and 1 which is the chance of the accurate language being associated with a situation
    - random_seed: Integer - Seed for the random generator used to pick which situations language is not accurate
    - language_random_seed: Integer - Seed for the random chooser which picks the incorrect object to make language for if the situation language should be inaccurate

### Pursuit Integrated Learner Experiment
- include_attributes: Boolean (Optional: True)
- include_relations: Boolean (Optional: True)
- pursuit_job_limit: Int (Optional: 8)
