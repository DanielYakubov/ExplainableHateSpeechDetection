import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

class HateSpanClsDataset(Dataset):
    """creating the dataset needed to run huggingface trainer on data"""

    def __init__(self, encodings, labels):
        """
        Args
            :param csv_file_path: the path of the data needed to be transformed into a dataset
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item