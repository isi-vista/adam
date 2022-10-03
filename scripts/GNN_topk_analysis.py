import argparse
import logging
from os import makedirs
from pathlib import Path
from shutil import rmtree, copytree
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import yaml

def parse_in_domain_objects(input_dir: Path) -> Sequence:
    in_domain_objects = set()
    for feature_file_path in sorted(input_dir.glob("situation*")):
        with open(feature_file_path / 'description.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            object = description_yaml['language'].rsplit()[-1]
            in_domain_objects.add(object)
    return sorted(in_domain_objects)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get confusion matrix for GNN decode"
    )
    parser.add_argument('--input-dir', type=Path, help="Inpuut curriculum directory to read GNN decodes from")
    parser.add_argument('--output-dir', type=Path, help="Output directory to write top GNN decode matrix and data to")
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        required=False,
        help='Force overwrite of target directory. By default, the script exits with an error if there already exists '
             'a directory at the target destination.'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    force_overwrite: bool = args.force

    if not input_dir.exists():
        logging.warning("Input directory does not exist")
        raise FileNotFoundError(str(input_dir))

    if output_dir.is_dir():
        if not force_overwrite:
            logging.warning("There already exists a directory in the target location")
            raise FileExistsError(str(output_dir))
    else:
        makedirs(output_dir)

    in_domain_objects = parse_in_domain_objects(input_dir)
    object_counts = {in_domain_object: 0 for in_domain_object in in_domain_objects}
    gnn_rank_matrix = [[0 for _ in range(len(in_domain_objects))] for _ in range(len(in_domain_objects))]
    for feature_file_path in sorted(input_dir.glob("situation*")):
        with open(feature_file_path / 'description.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            expected_object = description_yaml['language'].rsplit()[-1]
            object_counts[expected_object] += 1
        with open(feature_file_path / 'feature.yaml', encoding="utf-8") as feature_file:
            feature_yaml = yaml.safe_load(feature_file)
            gnn_objects = feature_yaml['objects'][0]['stroke_graph']['concept_names']
            for i in range(len(gnn_objects)):
                gnn_object = gnn_objects[i]
                if gnn_object == 'small_single_mug': gnn_object = 'mug'
                gnn_rank_matrix[in_domain_objects.index(expected_object)][in_domain_objects.index(gnn_object)] += len(gnn_objects) - i
    for object_id in range(len(gnn_rank_matrix)):
        gnn_rank_matrix[object_id] = [label_count / object_counts[in_domain_objects[object_id]] for label_count in arr[object_id]]
    df = pd.DataFrame(gnn_rank_matrix, in_domain_objects, in_domain_objects)
    sn.set(font_scale=0.9)
    sn.color_palette('colorblind')
    sn.set_context('paper')
    heatmap = sn.heatmap(df, annot=True, cbar=False)
    heatmap.set_xticklabels(labels=in_domain_objects, rotation=45, horizontalalignment='right')
    plt.xlabel('GNN Object Label Weight')
    plt.ylabel('True Object Label')
    plt.title('Top GNN Predictions per Object', size=18)
    plt.tight_layout()

    plt.savefig(output_dir / 'top_gnn_vs_description.png')
    df.to_csv(output_dir / 'top_gnn_vs_description.csv')


if __name__ == '__main__':
    main()