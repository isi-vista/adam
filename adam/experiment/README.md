# Experimentation Setup

Our experiments can be influenced by a number of parameters. This guide is intended to list all potential parameters, their impacts on the relevant portion of the curriculum, and any conflicting settings.

## Generic Parameter Control
These parameters are independent of any specific experiment.
### Curriculum Params
- curriculum: String - The name of the curriculum to generate
    - Valid options: "m6-deniz", "each-object-by-itself", "pursuit", "m6-preposition", "m9-objects", "m9-attributes", "m9-relations", "m9-events", "m9-debug", "m9-complete", "m13-imprecise-size", "m13-imprecise-temporal", "m13-subtle-verb-distinction", "m13-object-restrictions", "m13-functionally-defined-objects", "m13-generics", "m13-complete", "m13-verbs-with-dynamic-prepositions", "m13-shuffled", "m13-relations", "m15-object-noise-experiments"
- pursuit-curriculum-params: Namespace - The configuration for the pursuit specific curriculum
- use-path-instead-of-goal: Boolean (Optional) - If true some curriculums use a path indication rather than a goal
- num_samples: Integer (Optional) - The number of each specific situation type to instantiate
- num_noise_objects: Integer (Optional) - The number of noise objects in a situation

### Experiment Params
- experiment: String - The experiment name
- debug_log_directory: Path (Optional) - The path to log associated debug information, in this case it enables debug graph logging
- language_mode: String - The language the experiment should use. This string value converts to an enumerated value
    - Valid options: "English", "Chinese"
- experiment_group_dir: Path - The path where the experiment will be logged to
- learner: String - The name of the learner type to use
    - Valid options: "pursuit", "object-subset", "preposition-subset", "attribute-subset", "verb-subset", "integrated-learner", "integrated-learner-recognizer", "pursuit-gaze", "integrated-object-only"
- beam_size: Integer (Optional) - Used to constrain the search space for learner algorithms

- pre_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's output of a situation before the learn process is run on the new input.
- post_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's output of a situation after the learning process is ran on the new input.
- test_observer: namespace (Optional) - This namespace parameter is an Observer Params section. This observer reviews the learner's performance on the test situations.
- hypothesis_log_dir: Path (Optional) - This path is a directory to log internal learner hypotheses into
- log_hypothesis_every_n_steps: Integer (Optional) - The number of situations to observe before the system logs the internal state of the experiment
- log_learner_state: Boolean (Optional) - If true, save the internal trained learner state in a pickle.
- starting_point: Integer (Optional) - Useful when debugging to start training from a specific situation number in the curriculum. Can not be used if `resume_from_latest_logged_state` is true.
- point_to_log: Integer (Optional) - Useful debugging tool causes a specific log of hypotheses, learner, & observer states (future functionality for Observers)
- learner_state_path: Path (Optional) - Provide a path to a pickled learner file to load to continue an experiment with. Used commonly in combination with `starting_point`. Can not be used if `resume_from_latest_logged_state` is true.
- resume_from_latest_logged_state: Boolean (Optional) - Restart an experiment from the most recent logged state that can be located in the directory structure
- debug_learner_pickling: Boolean (Optional) - If true, log additional debug information for the learner when pickling

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
### Subset
None

### Propose But Verify
- graph_match_confirmation_threshold: Float - A value between 0 and 1 which indicates how much a new hypothesis must match a previous pattern to count as a successful match.
- random_seed: Integer - A seed for the RandomChooser which will make the random decisions between possible meanings

### Cross-Situational
< Not Yet Implemented >

### Pursuit
- learning_factor: Float - The learning factor rate
- graph_match_confirmation_threshold: Float - A value between 0 and 1 which indicates how much a new hypothesis must match a previous pattern to count as a successful match.
- lexicon_entry_threshold: Float - A value between 0 and 1 which indicates what probability a concept should be allowed into the 'known' lexicon
- smoothing_parameter: Float - A value between 0 and 1 which is generally small which smooths the probabilities over time.

## Specific Experiment Details
Any experiment specific parameters that are built into the program are explained below.

### Gaze Ablation

### Object Language Ablation
- object_learner_type: String - Defines which specific learning algorithm to implement for Objects
    - Valid options: "subset", "pbv", "pursuit"
- learner_params: Namespace - Defines a set of parameters to configure the different learner types
- train_curriculum: Namespace - Defines a set of parameters to configure the training curriculum for this experiment
    - accurate_language_percentage: Float - A value between 0 and 1 which is the chance of the accurate language being associated with a situation
    - random_seed: Integer - Seed for the random generator used to pick which situations language is not accurate
    - language_random_seed: Integer - Seed for the random chooser which picks the incorrect object to make language for if the situation language should be inaccurate
