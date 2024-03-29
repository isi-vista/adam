import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

import requests
import yaml
from tqdm import tqdm

API_ROUTE = "/instanceseg"
DESCRIPTION_FILENAME = "description.yaml"
INFO_FILENAME = "info.yaml"
COLOR_SEED = 42
NP_RNG = np.random.default_rng(COLOR_SEED)
LABEL_COLORS = NP_RNG.uniform(0, 255, size=(50, 3))
STEGO_MODEL = "stego"
MASK_RCNN_MODEL = "rcnn"


# this should take a sequence of 4-tuples, but that form is not very convenient to use with the raw
# JSON response
def with_maximal_area(boxes: Sequence[Sequence[float]]) -> Sequence[int]:
    """
    Given some bounding boxes, identify those with maximal area.

    Parameters:
        boxes: A sequence of bounding boxes (x0, y0, x1, y1).

    Returns:
        A sequence of indices into :param:`boxes`.
    """
    threshold = 0.1
    indices = []
    max_area = 0.0
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        area = abs((x1 - x0) * (y1 - y0))
        if max_area - threshold <= area <= max_area + threshold:
            indices.append(idx)
        elif area > max_area + threshold:
            max_area = area
            indices = [idx]

    return indices


def with_maximal_size(masks: Sequence[Sequence[Sequence[bool]]]) -> Sequence[int]:
    """
    Given some image masks, identify those with maximal size.

    The size is the total number of pixels included in the mask, i.e. the number of True base
    elements.

    Parameters:
        masks: A sequence of masks.

    Returns:
        A sequence of indices into :param:`masks`.
    """
    indices = []
    max_pixels = 0
    for idx, mask in enumerate(masks):
        n_pixels_included = sum(sum(int(elem) for elem in row) for row in mask)
        if n_pixels_included == max_pixels:
            indices.append(idx)
        elif n_pixels_included > max_pixels:
            max_pixels = n_pixels_included
            indices = [idx]

    return indices


def identify_background(response: Mapping[str, Any]) -> int:
    """
    Identify the background mask index for the given response.

    This is a hack:

    1. We find those masks with the biggest corresponding bounding box area, i.e. masks whose boxes
       are not smaller than those of any other mask.
    2. From those masks found in (1), we find the masks containing a maximal number of pixels, i.e.
       masks where no other mask contains more pixels.
    3. Finally, we pick an arbitrary mask from those in (2).

    Parameters:
        response:
            The response from the segmentation server.

    Returns:
        The index of the mask corresponding to the background.
    """
    maximal_box_idcs = with_maximal_area(response["boxes"])
    maximal_mask_idcs = with_maximal_size(
        [response["masks"][i] for i in maximal_box_idcs]
    )
    return maximal_box_idcs[maximal_mask_idcs[0]]


def stego_postprocessing(data: Mapping[str, Any], img: Any) -> Any:
    # Comment so current unused data in the `data` map can be recalled in the future
    # boxes, label = data["boxes"], data["labels"]
    masks = np.array(data["masks"])

    beta = 0.6
    alpha = 1.0
    gamma = 0.0
    for idx, mask in enumerate(masks):
        r_map = np.zeros_like(mask).astype(np.uint8)
        g_map = np.zeros_like(mask).astype(np.uint8)
        b_map = np.zeros_like(mask).astype(np.uint8)

        # mask color assignment
        color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        r_map[mask == 1], g_map[mask == 1], b_map[mask == 1] = color

        # combine masks
        segmentation_map = np.stack([r_map, g_map, b_map], axis=2)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # instance mask
        cv2.addWeighted(img, alpha, segmentation_map, beta, gamma, img)

    return img


def rcnn_postprocessing(data: Mapping[str, Any], img: Any) -> Any:
    # 3D array: (N, W, H) where N is number of masks
    masks = np.asarray(data["masks"], dtype=np.uint8)

    # Check for and warn about overlapping masks. We warn because this is an edge case
    # where we don't have a good solution. Currently, we deal with it using an arbitrary hack
    # of "prefer the later mask", which might give weird results sometimes.
    for i in range(len(masks)):  # pylint: disable=consider-using-enumerate
        for j in range(i + 1, len(masks)):  # pylint: disable=consider-using-enumerate
            if np.any(masks[i] & masks[j]):
                logging.warning(
                    f"Warning: Masks {i} and {j} overlap. Preferring later mask {j}."
                )

    # Compute pixelwise segmentation/mask IDs
    #
    # That is, an input-image-sized array (shape (W, H)) where each entry is either 0 (meaning
    # not in any mask) or a positive integer identifying which of the `N` output masks this pixel
    # belongs to. When a pixel lies in the overlap of several masks, we arbitrarily take the
    # highest-numbered mask.
    segmentation_ids = (
        np.max(
            masks
            * (1 + np.arange(masks.shape[0], dtype=masks.dtype))[
                :, np.newaxis, np.newaxis
            ],
            axis=0,
        )
        if masks.size > 0
        else np.zeros(img.shape[:2], dtype=np.uint32)
    )
    # Create segmentation mask image by assigning random colors to each mask
    rng = np.random.default_rng(COLOR_SEED)
    segment_colors = np.zeros((len(masks) + 1, 3), dtype=np.uint8)
    segment_colors[1:, :] = rng.uniform(0, 256, size=(len(masks), 3)).astype(np.uint8)
    return np.take(segment_colors, indices=segmentation_ids, axis=0)


def main() -> None:
    """Process a given curriculum directory for instance segmentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-curriculum",
        type=Path,
        required=True,
        help="The path to the base curriculum to use for segmentation queries.",
    )
    parser.add_argument(
        "--save-to",
        type=Path,
        required=True,
        help="The path to copy the curriculum to with instance segmentation. This will overwrite a curriculum.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[STEGO_MODEL, MASK_RCNN_MODEL],
        default=STEGO_MODEL,
        help="The instance segmentation model to use.",
    )
    parser.add_argument(
        "--api",
        type=str,
        default="http://saga03.isi.edu:5001",
        help="The server hostname & port to post API requests to.",
    )
    args = parser.parse_args()
    # Paths are converted here explicitly for type hints within IDEs
    base_curriculum = Path(args.base_curriculum)

    if not base_curriculum.is_dir():
        raise NotADirectoryError(f"{base_curriculum} is not a directory.")

    output_cur_path = Path(args.save_to)
    output_cur_path.mkdir(exist_ok=True, parents=True)

    logging.info("Beginning Curriculum Processing")
    # ugly hack to sort in 0, 1, 2, ... order
    for idx, sit_dir in tqdm(
        list(
            enumerate(
                sorted(
                    base_curriculum.glob("situation_*"),
                    key=lambda x: int(x.name.split("_")[1]),
                )
            )
        ),
        desc="Curriculum situation dir processing",
    ):
        sit_out_dir = output_cur_path / f"situation_{idx}"
        sit_out_dir.mkdir(parents=True, exist_ok=True)

        for idy, rgb_file in enumerate(sorted(sit_dir.glob("rgb_*.png"))):
            with open(rgb_file, "rb") as f:
                im_bytes = f.read()

            im_b64 = base64.b64encode(im_bytes).decode("utf8")
            headers = {"Content-type": "application/json", "Accept": "text/plain"}
            payload = json.dumps({"image": im_b64, "segmentation_type": args.model})
            resp = requests.post(f"{args.api}{API_ROUTE}", data=payload, headers=headers)

            orig_img = cv2.imread(str(rgb_file))
            data = resp.json()
            # Hack: If we're using STEGO, we identify the background mask's index (STEGO won't tell
            # us) then delete the data for that mask. This should cause the background mask's area
            # to be colored black, which is necessary for color refinement to work properly. See
            # adam_preprocessing/color_refinement.py.
            if args.model == STEGO_MODEL:
                background_idx = identify_background(data)
                # Ignore labels, linear_labels since those are empty
                logging.debug(
                    "Deleting background mask data at idx %d from situation %d image %d...",
                    background_idx,
                    idx,
                    idy,
                )
                # Only update masks and boxes, because the background we identified was one of the
                # unsupervised/clustering-based mask outputs. The linear layer may identify
                # different masks/boxes.
                for key in ["masks", "boxes"]:
                    if data[key]:
                        logging.debug(
                            "Removing index %d from response key %s", background_idx, key
                        )
                        del data[key][background_idx]

            if args.model == STEGO_MODEL:
                img = rcnn_postprocessing(data, orig_img)
            elif args.model == MASK_RCNN_MODEL:
                img = rcnn_postprocessing(data, orig_img)
            else:
                raise RuntimeError("Post-process undefined for this segmentation type.")

            cv2.imwrite(str(sit_out_dir / f"semantic_{idy}.png"), img)

    with open(base_curriculum / INFO_FILENAME, encoding="utf-8") as info_file:
        info = yaml.safe_load(info_file)

    info["preprocessing"] = {"segmentation": True, "segmentation_type": args.model}

    with open(output_cur_path / INFO_FILENAME, "w", encoding="utf-8") as info_file:
        yaml.dump(info, info_file)


if __name__ == "__main__":
    main()
