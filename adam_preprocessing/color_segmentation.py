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
from typing import Dict, Optional, Tuple, Union

import cv2
import matlab.engine
import numpy as np

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


def colors_to_index_np(
    color_img: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Given an RGB image, convert it to an indexed-color image.

    This implementation uses Numpy. It should be much faster but might not work in all cases.

    Parameters:
        color_img: A NumPy array of colors having shape (H, W, 3).

    Returns:
        At [0], a NumPy integer array of shape (H, W) where each value is an index.
            [1]: the index to color mapping.
            [2]: the array of unique colors of shape (N, 3) where N is the number of unique colors.
    """
    unique, inv = np.unique(color_img.reshape(-1, 3), axis=0, return_inverse=True)
    return inv.reshape(color_img.shape[0:2]), {k: unique[k] for k in inv}, unique


def colors_to_index_hack(
    color_img: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """
    Given an RGB image, convert it to an indexed-color image.

    This is a quick implementation, so it may be slow. I did a quick search for a similar function
    in cv2 and didn't find one.

    Parameters:
        color_img: A NumPy array of colors having shape (H, W, 3).

    Returns:
        At [0], a NumPy integer array of shape (H, W) where each value is an index.
            [1]: the index to color mapping.
            [2]: the array of unique colors of shape (N, 3) where N is the number of unique colors.
    """
    # Index colors
    color_to_index = {}
    k = 0
    for i in range(color_img.shape[0]):
        for j in range(color_img.shape[1]):
            color = color_img[i, j]
            if color not in color_to_index:
                color_to_index[color_img[i, j]] = k
                k += 1

    # Convert to indexed image
    indexed_image = np.zeros(color_img.shape[0:2], dtype=np.int)
    for i in range(color_img.shape[0]):
        for j in range(color_img.shape[1]):
            indexed_image[i, j] = color_to_index[color_img[i, j]]

    unique = np.stack([key for key in color_to_index.keys()], axis=0)
    return indexed_image, {v: k for k, v in color_to_index.items()}, unique


def with_random_colors(
    indexed_img: np.ndarray,
    *,
    n_colors: Optional[int] = None,
    color_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Given an indexed-color image, apply random colors to it.

    This is a quick implementation, so it may be slow. I did a quick search for a similar function
    in cv2 and didn't find one.

    Parameters:
        indexed_img: A NumPy array of shape (H, W) representing an indexed-color image.
        n_colors:
            The number of colors to use. Can be calculated from the indexed image; passing this just
            saves time.
        color_seed:
            The seed used to generate random colors.

    Returns:
        A color image with random colors, being a NumPy integer array of shape (H, W, 3).
    """
    if not n_colors:
        n_colors = np.max(indexed_img) + 1
    rng = np.random.default_rng(color_seed)
    random_colors = rng.uniform(0, 256, size=(n_colors, 3)).astype(np.uint8)
    return np.take(random_colors, indices=indexed_img, axis=0)


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
