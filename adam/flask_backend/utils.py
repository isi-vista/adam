from base64 import b64encode
from pathlib import Path
import re
from typing import Sequence, Union

from adam.paths import POST_LEARN_FILE_NAME


def get_image_data(image_file: Union[Path, str]) -> str:
    with open(image_file, "rb") as fp:
        image_data = b64encode(fp.read()).decode("utf-8")
    return image_data


def retrieve_relevant_files(situation_dir: Path) -> Sequence[Path]:
    matcher = re.compile(
        f"((rgb__[0-9]*.png)|(rgb_[0-9]*.png)|(id_rgb_[0-9]*.png)|(stroke_[0-9]*_[0-9]*.png)|"
        f"(stroke_graph_*.png)|({POST_LEARN_FILE_NAME}))$"
    )
    return [file for file in situation_dir.iterdir() if matcher.match(file.name)]
