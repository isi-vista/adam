"""
Perform color segmentation on an RGB image.

This uses a modified version of the MATLAB code from:

"Robust Image Segmentation Using Contour-guided Color Palettes" by Xiang Fu, Chien-Yi Wang,
Chen Chen, Changhu Wang and C.-C. Jay Kuo, in ICCV 2015.

Code used is as found here: https://github.com/fuxiang87/MCL_CCP
"""
from argparse import ArgumentParser
import logging
import os.path
from pathlib import Path
from typing import Tuple, Union

import cv2
import matlab.engine
import numpy as np

from mask_image import colors_to_index_np, with_random_colors

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)


def do_color_segmentation(imgpath: Union[Path, str]) -> np.ndarray:
    """
    Get raw color segmentation output from Matlab.

    Note that stroke extraction is performed on the object segmentation image, i.e. the
    semantic_j.png file. It is *not* done on the raw RGB image.

    Processing may fail for some images. For such images we raise a ValueError.

    Parameters:
        imgpath: The path to the image.

    Return:
         The output is an array [H, W, 3] where each [i, j, :] entry is a color.
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(file_dir), nargout=0)
    try:
        out = eng.CCP_seg(str(imgpath), nargout=1)
    except matlab.engine.MatlabExecutionError as e:
        raise ValueError(f"Could not properly color-segment image {imgpath}") from e
    eng.close()
    return (np.array(out) * 255).astype(np.uint8)


def segment_rgb_file(
    *,
    rgb: Path,
    save_with_proper_colors_to: Path,
    save_with_random_colors_to: Path,
    color_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the path to an RGB image, color-segment it saving to the given output path.

    Parameters:
        rgb: Path to an RGB image.
        save_with_proper_colors_to:
            Path where the aggregate-colors color-segmented output file should be saved.
        save_with_random_colors_to:
            Path where the random-colors color-segmented output file should be saved.
        color_seed: Seed to use when choosing random colors.
        
    Returns:
        The proper and random color segmentations in that order.
    """
    segmentation = do_color_segmentation(rgb)
    cv2.imwrite(str(save_with_proper_colors_to), cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB))

    indexed_color_segmentation, _index_to_color, unique = colors_to_index_np(segmentation)

    randomized_color_segmentation = with_random_colors(
        indexed_color_segmentation, n_colors=len(unique), color_seed=color_seed
    )
    cv2.imwrite(str(save_with_random_colors_to), randomized_color_segmentation)

    return segmentation, randomized_color_segmentation


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("rgb_img_path", type=Path, help="The RGB image to be segmented.")
    parser.add_argument(
        "save_stroke_extraction_seg_to",
        type=Path,
        help="Where to save the stroke-extraction-friendly color segmentation image. This "
        "identifies the regions by random colors rather than using any kind of color statistic for "
        "that region of the image.",
    )
    parser.add_argument(
        "save_color_seg_to",
        type=Path,
        help="Where to save the full-color segmentation image. This uses the aggregate color for "
        "each region.",
    )
    parser.add_argument(
        "--color_seed",
        type=int,
        default=42,
        help="The seed used to recolor the segmentation image when making it "
        "stroke-extraction-friendly.",
    )
    args = parser.parse_args()

    segment_rgb_file(
        rgb=args.rgb_img_path,
        save_with_proper_colors_to=args.save_color_seg_to,
        save_with_random_colors_to=args.save_stroke_extraction_seg_to,
        color_seed=args.color_seed,
    )


if __name__ == "__main__":
    main()
