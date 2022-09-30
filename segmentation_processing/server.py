"""Segementation processing server."""
import base64
from http import HTTPStatus
from io import BytesIO
import logging
import os
from typing import Any

from PIL import Image
from flask import Flask, abort, jsonify, request
from flask_cors import CORS
from gunicorn import glogging
from gunicorn.config import Config
import torch.multiprocessing
import torchvision
from torchvision.transforms import transforms

from mask_rcnn.utils import inference_res as rcc_inference_res
from stego.inf_utils import inference_res as stego_inference_res
from stego.train_segmentation import LitUnsupervisedSegmenter  # type: ignore

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

STEGO_MODEL_PATH = f"model{os.sep}cocostuff27_vit_base_5.ckpt"
LOGGING_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"

transform = transforms.Compose([transforms.ToTensor()])


class GunicornLogger(glogging.Logger):  # type: ignore
    """Logger for Gunicorn."""

    def __init__(self, cfg: Config) -> None:
        """Extended init function setting logging format."""
        super().__init__(cfg)
        self.access_log.addHandler(logging.StreamHandler())
        formatter = logging.Formatter(LOGGING_FORMAT)
        for handler in self.access_log.handlers:
            handler.setFormatter(formatter)


@app.route("/instanceseg", methods=["POST"])
def mask() -> Any:
    """Route for instance segmentation processing.

    Returns:
        A JSON response.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not request.json:
        abort(HTTPStatus.BAD_REQUEST)

    segmentation_type = request.json.get("segmentation_type", "stego")

    linear_masks = None
    linear_boxes = None
    labels = None

    img = Image.open(BytesIO(base64.b64decode(request.json["image"])))
    image = transform(img)
    image = image.unsqueeze(0).to(device)

    if segmentation_type == "rcnn":
        rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91
        )
        rcnn_model.to(device=device).eval()

        cluster_masks, cluster_boxes, labels = rcc_inference_res(image, rcnn_model, 0.5)
    elif segmentation_type == "stego":
        stego_model = LitUnsupervisedSegmenter.load_from_checkpoint(STEGO_MODEL_PATH)
        stego_model.to(device=device).eval()

        linear_masks, cluster_masks, linear_boxes, cluster_boxes = stego_inference_res(
            image, stego_model
        )
        linear_masks = linear_masks.tolist()
        cluster_masks = cluster_masks.tolist()
        linear_boxes = linear_boxes.tolist()
        cluster_boxes = cluster_boxes.tolist()
    else:
        return abort(HTTPStatus.BAD_REQUEST)

    payload = dict(
        masks=cluster_masks,
        linear_masks=linear_masks,
        boxes=cluster_boxes,
        linear_boxes=linear_boxes,
        labels=labels,
        linear_labels=None,
    )

    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, load_dotenv=False)
