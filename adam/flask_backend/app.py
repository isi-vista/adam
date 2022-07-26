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
from utils import get_image_data, retrieve_relevant_files

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["JSON_SORT_KEYS"] = False
app.config["LEARNER_DIR_CONTENTS"] = {}


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


@app.route("/api/get_image", methods=["GET"])
def get_image() -> Any:
    if not request.args:
        abort(HTTPStatus.BAD_REQUEST)

    learner = request.args.get("learner", default="")
    training_curriculum = request.args.get("training_curriculum", default="")
    testing_curriculum = request.args.get("testing_curriculum", default="")
    scene_number = int(request.args.get("scene_number", "")) - 1
    situation_dir = f"situation_{scene_number}"
    image_file = request.args.get("image_file", default="")
    if (
        not learner
        or not training_curriculum
        or not testing_curriculum
        or not situation_dir
    ):
        abort(HTTPStatus.BAD_REQUEST)

    if any(
        "/" in arg
        for arg in (
            learner,
            training_curriculum,
            testing_curriculum,
            situation_dir,
            image_file,
        )
    ):
        abort(HTTPStatus.BAD_REQUEST)
    image_path = (
        LEARNERS_DIR
        / learner
        / EXPERIMENTS_DIR_NAME
        / training_curriculum
        / EXPERIMENTS_TESTING_DIR_NAME
        / testing_curriculum
        / situation_dir
        / image_file
    ).resolve()
    if not is_relative_to(image_path, DATA_DIR):
        return {"message": "Directory out of range"}, HTTPStatus.BAD_REQUEST

    if (
        learner not in app.config["LEARNER_DIR_CONTENTS"]
        or training_curriculum not in app.config["LEARNER_DIR_CONTENTS"][learner]
        or testing_curriculum
        not in app.config["LEARNER_DIR_CONTENTS"][learner][training_curriculum]
        or situation_dir
        not in app.config["LEARNER_DIR_CONTENTS"][learner][training_curriculum][
            testing_curriculum
        ]
        or image_file
        not in app.config["LEARNER_DIR_CONTENTS"][learner][training_curriculum][
            testing_curriculum
        ][situation_dir]
    ):
        return {"message": "Invalid file"}, HTTPStatus.BAD_REQUEST
    return {"image_data": get_image_data(image_path)}


@app.route("/api/poll_changes", methods=["GET"])
def poll_changes() -> Any:
    if not request.args:
        abort(HTTPStatus.BAD_REQUEST)
    learner = request.args.get("learner", default="")
    training_curriculum = request.args.get("training_curriculum", default="")
    testing_curriculum = request.args.get("testing_curriculum", default="")
    if not learner or not training_curriculum or not testing_curriculum:
        abort(HTTPStatus.BAD_REQUEST)

    if any("/" in arg for arg in (learner, training_curriculum, testing_curriculum)):
        abort(HTTPStatus.BAD_REQUEST)
    curriculum_dir = (
        LEARNERS_DIR
        / learner
        / EXPERIMENTS_DIR_NAME
        / training_curriculum
        / EXPERIMENTS_TESTING_DIR_NAME
        / testing_curriculum
    ).resolve()

    if not is_relative_to(curriculum_dir, LEARNERS_DIR):
        return {"message": "Directory out of range"}, HTTPStatus.BAD_REQUEST

    if (
        learner not in app.config["LEARNER_DIR_CONTENTS"]
        or training_curriculum not in app.config["LEARNER_DIR_CONTENTS"][learner]
        or testing_curriculum
        not in app.config["LEARNER_DIR_CONTENTS"][learner][training_curriculum]
        or not curriculum_dir.is_dir()
    ):
        return {"status": False, "message": "Invalid curriculum"}

    ref_experiments = app.config["LEARNER_DIR_CONTENTS"][learner][training_curriculum][
        testing_curriculum
    ]
    new_decodes = []
    specific_updates = []
    for situation_dir in curriculum_dir.iterdir():
        if "situation" in situation_dir.name and situation_dir.is_dir():
            if situation_dir.name not in ref_experiments:
                ref_experiments[situation_dir.name] = set()
            new_files = (
                set([file.name for file in retrieve_relevant_files(situation_dir)])
                - ref_experiments[situation_dir.name]
            )
            if new_files:
                new_decodes.append(
                    f"New files available for situation {int(situation_dir.name.rsplit('_')[-1]) + 1}"
                )
                specific_updates.append(
                    f"Situation {int(situation_dir.name.rsplit('_')[-1]) + 1}:"
                )
                for file in sorted(new_files):
                    specific_updates.append(f"\t{file}")
                ref_experiments[situation_dir.name].update(new_files)
    return (
        {
            "status": True,
            "message": "\n".join(new_decodes),
            "sub_message": "\n".join(specific_updates),
        }
        if new_decodes or specific_updates
        else {"status": False}
    )


def init_dir_contents() -> None:
    for learner in LEARNERS_DIR.iterdir():
        if learner.is_dir() and (learner / EXPERIMENTS_DIR_NAME).is_dir():
            app.config["LEARNER_DIR_CONTENTS"][learner.name] = {}
            for training_curriculum in (learner / EXPERIMENTS_DIR_NAME).iterdir():
                if (
                    training_curriculum.is_dir()
                    and (training_curriculum / EXPERIMENTS_TESTING_DIR_NAME).is_dir()
                ):
                    app.config["LEARNER_DIR_CONTENTS"][learner.name][
                        training_curriculum.name
                    ] = {}
                    for testing_curriculum in (
                        training_curriculum / EXPERIMENTS_TESTING_DIR_NAME
                    ).iterdir():
                        if testing_curriculum.is_dir():
                            app.config["LEARNER_DIR_CONTENTS"][learner.name][
                                training_curriculum.name
                            ][testing_curriculum.name] = {}
                            for situation_dir in testing_curriculum.iterdir():
                                if (
                                    "situation" in situation_dir.name
                                    and situation_dir.is_dir()
                                ):
                                    app.config["LEARNER_DIR_CONTENTS"][learner.name][
                                        training_curriculum.name
                                    ][testing_curriculum.name][situation_dir.name] = set(
                                        file.name
                                        for file in retrieve_relevant_files(situation_dir)
                                    )


if __name__ == "__main__":
    init_dir_contents()
    app.run(debug=True, load_dotenv=False)
# else:
#     gunicorn_logger = logging.getLogger("gunicorn.error")
#     logger.level = gunicorn_logger.level
