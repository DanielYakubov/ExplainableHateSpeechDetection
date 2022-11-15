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
        self.data = pd.read_csv(tsv_file_path, delimiter='\t')
        self.spans = self.data["span"].to_list()
        self.a_s = self.data["span_label"].to_list()
        self.labels = self.data["post_hs_label"].to_list()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        assert idx < len(self), "index must be in range"
        span = self.spans[idx]
        label = self.labels[idx]
        a_s = self.a_s[idx] 
        return {"span": span, "a_s": a_s, "label": label}

    def update(self, idx, point, new_val):
        assert point in ["span", "label", "a_s"], "not a valid attribute"
        assert idx < len(self), "index must be in range"
        if point == "span":
            self.spans[idx] = new_val
        elif point == "label":
            self.labels[idx] = new_val
        elif point == "a_s":
            self.a_s = new_val
    

def create_cls_dataset(tsv_file_path: str) -> HateSpanClsDataset:
    """
    create a HateSpanDataset object given a .tsv filepath. This object can be used to fine tune a BERT model
    :param tsv_file_path: the path to the tsv file containing the data
    :return: a HateSpanDataset object
    """
    return HateSpanClsDataset(tsv_file_path)
