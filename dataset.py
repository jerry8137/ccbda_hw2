import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import os

import ipdb

class brainDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.parse_file()

    def parse_file(self):
        for root, dirs, files in sorted(os.walk(self.root)):
            for name in sorted(files):
                print(os.path.join(root, name))

def test():
    dataset = brainDataset(root='./data', mode='unlabeled')

if __name__=='__main__':
    test()
