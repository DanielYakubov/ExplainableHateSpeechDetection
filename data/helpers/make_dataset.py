from turtle import pd
import os
import pandas as pd
import numpy as np
import torch

import logging

from torch.utils.data import Dataset

class HateSpanClsDataset(Dataset):
    """creating the dataset needed to run huggingface trainer on data"""

    def __init__(self, tsv_file_path):
        """
        Args
            :param csv_file_path: the path of the data needed to be transformed into a dataset
        """
        self.data = pd.read_csv(tsv_file_path)
        self.spans = self.data["span"]
        self.labels = self.data["post_hs_label"]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        span = self.spans[idx]
        label = self.labels[idx]
        return {"Span": span, "Label": label}
    

def create_cls_dataset(tsv_file_path: str) -> HateSpanClsDataset:
    """
    create a HateSpanDataset object given a .tsv filepath. This object can be used to fine tune a BERT model
    :param tsv_file_path: the path to the tsv file containing the data
    :return: a HateSpanDataset object
    """
    return HateSpanClsDataset(tsv_file_path)
