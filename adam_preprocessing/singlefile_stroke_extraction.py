"""
Code for doing stroke extraction on one file.
"""
from argparse import ArgumentParser
import logging
from pathlib import Path

from shape_stroke_extraction import Stroke_Extraction

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "seg_img_path",
        type=Path,
        help="The segmentation image to process.",
    )
    parser.add_argument(
        "rgb_img_path",
        type=Path,
        help="The RGB image to process.",
    )
    parser.add_argument(
        "save_output_to_dir",
        type=Path,
        help="Directory where output files should be saved.",
    )
    parser.add_argument(
        "--string_label",
        default="dummy",
        help="String label to use."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    Stroke_Extraction(
        segmentation_img_path=str(args.seg_img_path),
        rgb_img_path=str(args.rgb_img_path),
        debug_matlab_stroke_img_save_path=str(args.save_output_to_dir / "matlab_stroke_0.png"),
        stroke_img_save_path=str(args.save_output_to_dir / "stroke_0.png"),
        stroke_graph_img_save_path=str(args.save_output_to_dir / "stroke_graph_0.png"),
        features_save_path=str(args.save_output_to_dir / "feature.yaml"),
        obj_type=args.string_label,
        obj_view="-1",
        obj_id="-1",
    ).get_strokes()


if __name__ == "__main__":
    main()
