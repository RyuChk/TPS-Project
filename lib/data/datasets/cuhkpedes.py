import json
import os
import logging

import torch
from PIL import Image

import torch.utils.data


class CHUNKPEDESDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, ann_file: str, *,
                 use_onehot=True,
                 max_length=100,
                 transforms=None):
        self.root = root
        self.use_onehot = use_onehot
        self.max_length = max_length
        self.transforms = transforms

        self.img_dir = os.path.join(self.root, 'imgs')

        logging.debug('loading image annotations')
        dataset = json.load(open(ann_file, 'r'))
        self.dataset = dataset

    def __getitem__(self, i: int):
        data = self.dataset[i]

        img_path = data['file_path']
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
