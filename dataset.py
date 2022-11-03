from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from gaussian_blur import GaussianBlur

import os
from tqdm import tqdm


color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

default_transform = transforms.Compose([
    transforms.ToTensor(),
])

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1*64)),
    transforms.ToTensor(),
])


class brainDataset(Dataset):
    def __init__(self, root, mode, cache=True, gen=False):
        self.root = root
        self.mode = mode
        self.cache = cache
        self.gen = gen
        assert self.mode == 'test' or self.mode == 'unlabeled'
        self.root = os.path.join(self.root, self.mode)
        self.files = []
        self.labels = []

        self.parse_file()
        if cache:
            self.cache_image()

    def __getitem__(self, i):
        if self.cache:
            img = self.images[i]
        else:
            img = Image.open(self.files[i])
        if self.mode == 'unlabeled':
            if self.gen == True:
                return default_transform(img)
            return augment_transform(img), augment_transform(img)
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

    def cache_image(self):
        self.images = [Image.open(path).copy() for path in tqdm(self.files)]


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
