from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import os
from tqdm import tqdm


color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


class brainDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode == 'test' or self.mode == 'unlabeled'
        self.root = os.path.join(self.root, self.mode)
        self.files = []
        self.labels = []

        self.parse_file()

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        if self.mode == 'unlabeled':
            return default_transform(img), augment_transform(img)
        return default_transform(img), self.labels[i]

    def __len__(self):
        return len(self.files)

    def parse_file(self):
        for root, dirs, files in sorted(os.walk(self.root)):
            for name in sorted(files):
                file_path = (os.path.join(root, name))
                self.files.append(file_path)
                if self.mode == 'test':
                    label = int(root.split('/')[-1])
                    self.labels.append(label)


def test():
    test_dataset = brainDataset(root='./data', mode='test')
    unlabeled_dataset = brainDataset(root='./data', mode='unlabeled')
    for i in tqdm(test_dataset):
        pass
    for i in tqdm(unlabeled_dataset):
        pass

    print(test_dataset[0][0].shape)


if __name__ == '__main__':
    test()
