"""
Perform color segmentation refinement on an RGB image and pre-existing segmentation mask.
"""
from argparse import ArgumentParser
import logging
from pathlib import Path

import cv2
import numpy as np

from color_segmentation import (
    do_color_segmentation,
    colors_to_index_np,
    with_random_colors,
)

logger = logging.getLogger(__name__)


def refine_segmentation_simple(
    segmentation: np.ndarray,
    color_segmentation: np.ndarray,
) -> np.ndarray:
    """
    Refine an instance/object segmentation using the given color segmentation.

    Parameters:
        segmentation:
            The (H, W, 3) shaped color instance-segmentation image. Black is assumed to be
            "no object/mask."
        color_segmentation:
            The (H, W, 3) shaped color segmentation-by-color image.

    Returns:
        The color-refined segmentation image. That is, the segmentation-by-color image with the
        non-object parts colored black.
    """
    return np.where(
        segmentation != np.zeros([1, 1, 3]), color_segmentation, np.zeros([1, 1, 3])
    )


def refined_segmentations(color_segmentation: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """
    Refine an instance/object segmentation using the given color segmentation, producing N refined
    segmentations.

    Parameters:
        segmentation:
            The (H, W, 3) shaped color instance-segmentation image. Black is assumed to be
            "no object/mask."
        color_segmentation:
            The (H, W, 3) shaped color segmentation-by-color image.

    Returns:
        An array (N, H, W, 3) of color-refined segmentation images, one per object mask.
    """
    flat_segmentation_colors = segmentation.reshape(-1, 3)
    mask_colors = np.unique(
        flat_segmentation_colors[np.all(flat_segmentation_colors != 0, axis=1)], axis=0,
    )
    color_refined_segmentations = np.zeros([len(mask_colors), *segmentation.shape])
    for mask_id, mask_color in enumerate(mask_colors):
        mask = segmentation == mask_color[np.newaxis, np.newaxis, :]
        color_refined_segmentations[mask_id] = np.where(
            mask, color_segmentation, np.zeros([1, 1, 3])
        )

    return color_refined_segmentations


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("rgb_img_path", type=Path, help="The RGB image to be segmented.")
    parser.add_argument(
        "seg_img_path", type=Path, help="The segmentation image to be refined."
    )
    parser.add_argument(
        "--save_refined_segs_to",
        type=Path,
        default=None,
        help="Directory in which to save the refined segmentation images.",
    )
    parser.add_argument(
        "--save_combined_refined_seg_to",
        type=Path,
        default=None,
        help="File in which to save the combined refined segmentation image.",
    )
    parser.add_argument(
        "--use_random_colors",
        type=lambda x: x == "True",
        default=True,
        help="If true, use random colors when generating the refined segmentation mask.",
    )
    parser.add_argument(
        "--color_seed",
        type=int,
        default=42,
        help="The seed used to generate colors for the refined segmentation regions.",
    )
    args = parser.parse_args()

    segmentation = cv2.imread(str(args.seg_img_path))
    color_segmentation = do_color_segmentation(args.rgb_img_path)

    # Randomize color segmentation colors if needed.
    if args.use_random_colors:
        index, _index_to_color, unique = colors_to_index_np(color_segmentation)
        color_segmentation = with_random_colors(
            index, n_colors=len(unique), color_seed=args.color_seed
        )

    # Create and save the combined color segmentation.
    if args.save_combined_refined_seg_to is not None:
        cv2.imwrite(
            str(args.save_combined_refined_seg_to),
            refine_segmentation_simple(segmentation, color_segmentation),
        )

    # For each mask in the segmentation image, create a color-refined version.
    if args.save_refined_segs_to:
        color_refined_segmentations = refined_segmentations(
            color_segmentation, segmentation
        )

        for mask_id, color_refined_segmentation in enumerate(color_refined_segmentations):
            cv2.imwrite(
                str(args.save_refined_segs_to / f"color_refined_seg_{mask_id}.png"),
                color_refined_segmentation,
            )


if __name__ == "__main__":
    main()
