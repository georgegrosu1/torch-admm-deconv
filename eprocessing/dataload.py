from pathlib import Path
from typing import List
from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self,
                 x_source: Path,
                 y_source: Path,
                 transforms: List=None):

        self.x_source = x_source
        self.y_source = y_source
        self.transforms = transforms

        self.x_paths = list(x_source.glob("*"))
        self.y_paths = list(y_source.glob("*"))


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx: int):
        x_im = read_image(str(self.x_paths[idx]))
        y_im = read_image(str(self.y_paths[idx]))

        if self.transforms is not None:
            for transform in self.transforms:
                x_im, y_im = transform(x_im, y_im)

        return x_im, y_im