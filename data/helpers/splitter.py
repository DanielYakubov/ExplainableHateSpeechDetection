"""HateXplain has a standard split, we want to follow it."""

import csv
import json
import logging
from typing import Dict


def split(reference: Dict, filepath: str) -> None:
    """
    Writes a train, val, and test path for a given dataset according to an ID reference
    :param reference: a dictionary containing information about the standard split. the keys should be [train, test, val]
    :param filepath: the path to the tsv
    :return: none, writes files
    """
    with open(filepath) as data:
        csv_reader = csv.reader(data, delimiter='\t')
        output_fp = filepath[:-4]

        # old school, don't want to have too many with statements
        train_file = open(f"{output_fp}_train.tsv", "w")
        val_file = open(f"{output_fp}_val.tsv", "w")
        test_file = open(f"{output_fp}_test.tsv", "w")
        
        train_writer = csv.writer(train_file, delimiter='\t')
        val_writer = csv.writer(val_file, delimiter='\t')
        test_writer = csv.writer(test_file, delimiter='\t')
        
        header = next(csv_reader)
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

        for row in csv_reader:
            id, *args = row
            if id in reference['train']:
                train_writer.writerow(row)
            elif id in reference['val']:
                val_writer.writerow(row)
            elif id in reference['test']:
                test_writer.writerow(row)
            else:
                logging.warning(f"row: {row} not found in reference")

    train_file.close()
    val_file.close()
    test_file.close()
    
    

if __name__ == '__main__':
    with open("../datasets/divisions.json") as reference:
        reference = json.load(reference)
    split(reference=reference, filepath='../datasets/span_annotation.tsv')
    split(reference=reference, filepath='../datasets/original_data.tsv')