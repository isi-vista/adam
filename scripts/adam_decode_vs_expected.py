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
        with open(feature_file_path / 'post_decode.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            object = description_yaml['gold_language'].rsplit()[-1]
            in_domain_objects.add(object)
    return sorted(in_domain_objects)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get confusion matrix for GNN decode"
    )
    parser.add_argument('--input-dir', type=Path, help="Curriculum directory with ADAM decode files to read from")
    parser.add_argument('--output-dir', type=Path, help="Target directory to output confusion matrix and associated data to")
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

    in_domain_objects=parse_in_domain_objects(input_dir)
    confusion_matrix = [[0 for _ in range(len(in_domain_objects) + 1)] for _ in range(len(in_domain_objects))]
    none_count = 0
    correct_count = 0
    for feature_file_path in sorted(input_dir.glob("situation*")):
        with open(feature_file_path / 'post_decode.yaml', encoding="utf-8") as decode_file:
            decode_yaml = yaml.safe_load(decode_file)
            expected_object = decode_yaml['gold_language'].rsplit()[-1]
            if len(decode_yaml['output_language']) > 0:
                gnn_object = decode_yaml['output_language'][0]['raw_text'].rsplit()[-1]
                print(expected_object, gnn_object)
                confusion_matrix[in_domain_objects.index(expected_object)][in_domain_objects.index(gnn_object)] += 1
                if expected_object == gnn_object:
                    correct_count += 1
            else:
                none_count += 1
                confusion_matrix[in_domain_objects.index(expected_object)][-1] += 1


    df = pd.DataFrame(confusion_matrix, in_domain_objects, in_domain_objects + ['No Label'])
    sn.set(font_scale=0.9)
    sn.color_palette('colorblind')
    sn.set_context('paper')
    heatmap = sn.heatmap(df, annot=True, cbar=False)
    heatmap.set_xticklabels(labels=in_domain_objects + ['No Label'], rotation=45, horizontalalignment='right')
    plt.xlabel('ADAM Output Language')
    plt.ylabel('Expected Object Label')
    plt.title('Adam Decode vs Expected Label', size=18)
    plt.tight_layout()

    plt.savefig(output_dir / 'adam_decode_vs_description.png')
    df.to_csv(output_dir / 'adam_decode_vs_description.csv')
    print("Correctly labeled:", correct_count)
    print("% Correct:", correct_count/sum(sum(i) for i in arr))
    print("Objects with no label:", none_count)

if __name__ == '__main__':
    main()
