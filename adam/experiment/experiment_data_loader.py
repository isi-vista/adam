import os
from pathlib import Path
from typing import List
import yaml


def get_train_curricula(output_root: Path) -> List[str]:
    r"""
    Get list of training curricula from an experiment output directory.
    """
    return [dir for dir in os.listdir(output_root / "curriculum" / "train")]


def get_test_curricula(output_root: Path) -> List[str]:
    r"""
    Get list of test curricula from an experiment output directory.
    """
    return [dir for dir in os.listdir(output_root / "curriculum" / "test")]


def get_learners(output_root: Path) -> List[str]:
    r"""
    Get list of learners from an experiment output directory.
    """
    return [dir for dir in os.listdir(output_root / "learners")]


def load_situation_output(situation_output_path: Path):
    r"""
    Load all outputs given a path for the situation output
    """
    situation_outputs = {}
    for file_name in os.listdir(situation_output_path):
        if ".yaml" in file_name:
            # Load all yaml files
            with open(situation_output_path / file_name, "r") as file:
                situation_outputs[file_name.replace(".yaml", "")] = yaml.load(file)
        else:
            # Return paths for files, eg. png
            situation_outputs[file_name] = situation_output_path / file_name
    return situation_outputs


def load_curriculum_outputs(output_path: Path):
    r"""
    Load all learned situations given an output path that contains all situation outputs
    """
    outputs = {}
    for situation in os.listdir(output_path):
        # Get outputs and curriculum information for each situation
        outputs[situation] = load_situation_output(output_path / situation)
    return outputs


def load_train_output(output_root: Path, learner: str, training_curriculum: str):
    r"""
    Load all learned situations for a train curriculum
    """
    train_output_path = (
        output_root / "learners" / learner / "experiments" / training_curriculum
    )
    return load_curriculum_outputs(train_output_path)


def load_test_output(
    output_root: Path, learner: str, train_curriculum: str, test_curriculum: str
):
    r"""
    Load all learned situations for a test curriculum
    """
    test_output_path = (
        output_root
        / "learners"
        / learner
        / "experiments"
        / train_curriculum
        / "test_curriculums"
        / test_curriculum
    )
    return load_curriculum_outputs(test_output_path)
