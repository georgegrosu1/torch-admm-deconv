import cv2
import numpy as np

from typing import Tuple, List
from pathlib import Path


def add_blur_gaussian(img, k_shape=(17, 17), std=2.4):
    # add gaussian blurring
    return cv2.GaussianBlur(img, k_shape, std)


def add_noise_gaussian(img, mean=0, stdv=25):
    dst = np.zeros_like(img)
    noise = cv2.randn(dst, (mean, mean, mean), (stdv, stdv, stdv))
    # Noise overlaid over image
    return cv2.add(img, noise)


def get_dset_im_paths(txt_file: Path) -> Tuple[List[Path], List[Path]]:
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    y_paths = [txt_file.parent / x_p.split(' ')[0] for x_p in lines]
    x_paths = [txt_file.parent / x_p.split(' ')[1] for x_p in lines]
    return x_paths, y_paths
