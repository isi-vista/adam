from glob import glob
from http import HTTPStatus
from os.path import relpath
from typing import Any

import yaml

from itertools import chain
from flask import Flask, abort, request
from flask_cors import CORS

from adam.paths import (
    DATA_DIR,
    LEARNERS_DIR,
    SITUATION_DIR_NAME,
    POST_LEARN_FILE_NAME,
    EXPERIMENTS_DIR_NAME,
    EXPERIMENTS_TESTING_DIR_NAME,
    TRAINING_CURRICULUM_DIR,
    TESTING_CURRICULUM_DIR,
    is_relative_to,
)

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["JSON_SORT_KEYS"] = False


@app.route("/api/learners", methods=["GET"])
def get_all_learners() -> Any:
    """Get all learner configurations which are prepared."""

    return {
        "learner_types": sorted(
            [
                possible_dir.name
                for possible_dir in LEARNERS_DIR.iterdir()
                if possible_dir.is_dir()
            ]
        )
    }


@app.route("/api/training_curriculum", methods=["GET"])
def get_all_train_curriculum() -> Any:
    """Get all possible training curriculum."""
    return {
        "training_curriculum": sorted(
            [
                possible_dir.name
                for possible_dir in TRAINING_CURRICULUM_DIR.iterdir()
                if possible_dir.is_dir()
            ]
        ),
    }


@app.route("/api/testing_curriculum", methods=["GET"])
def get_all_test_curriculum() -> Any:
    """Get all available test curriculum."""
    return {
        "testing_curriculum": sorted(
            [
                possible_dir.name
                for possible_dir in TESTING_CURRICULUM_DIR.iterdir()
                if possible_dir.is_dir()
            ]
        )
    }


@app.route("/api/load_scene", methods=["GET"])
def get_scene() -> Any:
    """Get a specific scene from disk or the learner."""
    if not request.args:
        abort(HTTPStatus.BAD_REQUEST)

    learner = request.args.get("learner", default="")
    training_curriculum = request.args.get("training_curriculum", default="")
    testing_curriculum = request.args.get("testing_curriculum", default="")
    scene_number = int(request.args.get("scene_number", "")) - 1
    if (
        not learner
        or not training_curriculum
        or not testing_curriculum
        or not scene_number
    ):
        abort(HTTPStatus.BAD_REQUEST)
    if any("/" in arg for arg in (learner, training_curriculum, testing_curriculum)):
        abort(HTTPStatus.BAD_REQUEST)

    experiment_dir = (
        LEARNERS_DIR
        / learner
        / EXPERIMENTS_DIR_NAME
        / training_curriculum
        / EXPERIMENTS_TESTING_DIR_NAME
        / testing_curriculum
        / SITUATION_DIR_NAME.format(num=scene_number)
    ).resolve()
    if not is_relative_to(experiment_dir, LEARNERS_DIR):
        return {"message": "Directory out of range"}, HTTPStatus.BAD_REQUEST
    if not experiment_dir.exists():
        return {"message": "Selected configuration does not exist"}

    # This section below can be replaced with inference on a live model in the future
    if not (experiment_dir / POST_LEARN_FILE_NAME).exists():
        return {"message": "Learner has not decoded this scene"}

    with open(experiment_dir / POST_LEARN_FILE_NAME, encoding="utf-8") as yaml_file:
        post_learn = yaml.safe_load(yaml_file)

    return {
        "learner": learner,
        "train_curriculum": training_curriculum,
        "test_curriculum": testing_curriculum,
        "scene_number": scene_number,
        "scene_images": [
            relpath(path, DATA_DIR)
            for path in sorted(
                chain(
                    glob(f"{experiment_dir}/rgb__[0-9]*.png"),
                    glob(f"{experiment_dir}/id_rgb_[0-9]*.png"),
                )
            )
        ],
        "object_strokes": [
            relpath(path, DATA_DIR)
            for path in sorted(glob(f"{experiment_dir}/stroke_[0-9]*_[0-9]*.png"))
        ],
        "stroke_graph": [
            relpath(path, DATA_DIR)
            for path in sorted(glob(f"{experiment_dir}/stroke_graph_*.png"))
        ],
        "post_learning": post_learn,
        # "pre_learning": {},
        "message": None,
        # "gold_standard": {
        #     "language_with_scene": "a chair",
        # }
    }


if __name__ == "__main__":
    app.run(debug=True, load_dotenv=False)
# else:
#     gunicorn_logger = logging.getLogger("gunicorn.error")
#     logger.level = gunicorn_logger.level
