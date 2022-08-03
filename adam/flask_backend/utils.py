from base64 import b64encode
from pathlib import Path
from typing import Union


def get_image_data(image_file: Union[Path, str]) -> str:
    with open(image_file, "rb") as fp:
        image_data = b64encode(fp.read()).decode("utf-8")
    return image_data
