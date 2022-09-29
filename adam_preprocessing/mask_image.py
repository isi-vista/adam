"""
Code for working with mask images.

Mask images represent objects or regions using
"""
from typing import Dict, Optional, Tuple

import numpy as np


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
