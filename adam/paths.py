"""Paths used for data storage of experiments centralized so the UI can load from there."""
from pathlib import Path

TOP_LEVEL_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TOP_LEVEL_DIR / "data"
FONTS_DIR = DATA_DIR / "fonts"
ROBOTO_FILE = FONTS_DIR / "Roboto.ttf"
CURRICULUM_DIR = DATA_DIR / "curriculum"
TRAINING_CURRICULUM_DIR = CURRICULUM_DIR / "train"
TESTING_CURRICULUM_DIR = CURRICULUM_DIR / "test"
LEARNERS_DIR = DATA_DIR / "learners"

dirs = [
    DATA_DIR,
    FONTS_DIR,
    CURRICULUM_DIR,
    TRAINING_CURRICULUM_DIR,
    TESTING_CURRICULUM_DIR,
    LEARNERS_DIR,
]
for directory in dirs:
    directory.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_DIR_NAME = "experiments"
EXPERIMENTS_TESTING_DIR_NAME = "test_curriculums"
SITUATION_DIR_NAME = "situation_{num}"
PRE_LEARN_FILE_NAME = "pre_decode.yaml"
POST_LEARN_FILE_NAME = "post_decode.yaml"
GENERATION_YAML_DIR_NAME = "generation_yaml"
CURRICULUM_INFO_FILE = "info.yaml"
SITUATION_DESCRIPTION_FILE = "description.yaml"
SCENE_JSON = "scene.json"
FEATURE_YAML = "feature.yaml"
