import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
from enum import Enum

from utils.dset_utils import *


class Dset(Enum):
    GOPRO = 'gopro'
    HIDE = 'hide'
    REALBLUR = 'realblur'
    SIDD = 'sidd'
    RENOIR = 'renoir'


def get_train_test_txts(orig_p: Path) -> Tuple[List, List]:
    train_txts = list(orig_p.glob('*train.txt'))
    test_txts = list(orig_p.glob('*test.txt'))
    return train_txts, test_txts


def process_x_y_ims(x_ims: List, y_ims: List, min_stdv, max_stdv, sdir_x, sdir_y) -> None:
    for x_p, y_p in tqdm(zip(x_ims, y_ims), total=len(y_ims)):
        im_hash = get_rand_uuid()
        imx, imy = cv2.imread(str(x_p), cv2.IMREAD_COLOR), cv2.imread(str(y_p), cv2.IMREAD_COLOR)
        if max_stdv > 1 and max_stdv > min_stdv >= 0:
            r_stdv = np.random.randint(min_stdv, max_stdv)
            imx = add_noise_gaussian(imx, stdv=r_stdv)
        else:
            r_stdv = 0
        imx_p, imy_p = sdir_x / f'{im_hash}_awgn-{r_stdv}.png', sdir_y / f'{im_hash}.png'
        cv2.imwrite(imx_p, imx)
        cv2.imwrite(imy_p, imy)


def make_realblur_dset(orig: str,
                       save_dir_train_x: Path, save_dir_train_y: Path,
                       save_dir_test_x: Path, save_dir_test_y: Path,
                       min_noise_std: int, max_noise_std: int) -> None:

    train_txts, test_txts = get_train_test_txts(Path(orig))
    # Process train
    train_1_x, train_1_y = get_dset_im_paths(train_txts[0])
    train_2_x, train_2_y = get_dset_im_paths(test_txts[1])
    train_x, train_y = train_1_x + train_2_x, train_1_y + train_2_y
    print('\n\nProcessing train data')
    process_x_y_ims(train_x, train_y, min_noise_std, max_noise_std, save_dir_train_x, save_dir_train_y)

    # Process test
    test_1_x, test_1_y = get_dset_im_paths(test_txts[0])
    test_2_x, test_2_y = get_dset_im_paths(test_txts[1])
    test_x, test_y = test_1_x + test_2_x, test_1_y + test_2_y
    print('\n\nProcessing test data')
    process_x_y_ims(test_x, test_y, min_noise_std, max_noise_std, save_dir_test_x, save_dir_test_y)


def get_gopro_subset_im_paths(setdir: Path, subset: str = 'blur') -> tuple[list[Path], list[Path]]:
        x_im_paths, y_im_paths = [], []
        for sdir in setdir.iterdir():
            x_im_paths += list((sdir / subset).glob('*.png'))
            y_im_paths += list((sdir / 'sharp').glob('*.png'))

        return x_im_paths, y_im_paths


def get_hided_subset_im_paths(set_dir: Path) -> tuple[list[Path], list[Path]]:
    subset = set_dir.stem
    subset_txt_p = set_dir.parent / (subset + '.txt')
    with open(subset_txt_p, 'r') as f:
        lines = f.readlines()

    x_paths = [set_dir.parent / subset / line.replace('\n', '') for line in lines]
    y_paths = [set_dir.parent / 'GT' / line.replace('\n', '') for line in lines]

    return x_paths, y_paths


def get_sidd_subset_im_paths(set_dir: Path) -> tuple[list[Path], list[Path]]:
    png_files = list(set_dir.rglob("*.png"))
    x_paths = [png_file for png_file in png_files if 'NOISY' in png_file.stem]
    y_paths = [png_file for png_file in png_files if 'GT' in png_file.stem]

    return x_paths, y_paths


def make_gopro_dset(orig: str,
                    save_dir_train_x: Path, save_dir_train_y: Path,
                    save_dir_test_x: Path, save_dir_test_y: Path,
                    min_noise_std: int, max_noise_std: int) -> None:

    train_dirs = Path(f'{orig}/train')
    test_dirs = Path(f'{orig}/test')
    # Process train
    train_x, train_y = get_gopro_subset_im_paths(train_dirs, subset='blur')
    print('\n\nProcessing train data')
    process_x_y_ims(train_x, train_y, min_noise_std, max_noise_std, save_dir_train_x, save_dir_train_y)
    # Process test
    test_x, test_y = get_gopro_subset_im_paths(test_dirs, subset='blur')
    print('\n\nProcessing test data')
    process_x_y_ims(test_x, test_y, min_noise_std, max_noise_std, save_dir_test_x, save_dir_test_y)


def make_hide_dset(orig: str,
                   save_dir_train_x: Path, save_dir_train_y: Path,
                   save_dir_test_x: Path, save_dir_test_y: Path,
                   min_noise_std: int, max_noise_std: int) -> None:

    train_dirs = Path(fr'{orig}/train')
    test_dirs = Path(fr'{orig}/test')
    train_x, train_y = get_hided_subset_im_paths(train_dirs)
    train_x, train_y = train_x[::3], train_y[::3]
    print('\n\nProcessing train data')
    process_x_y_ims(train_x, train_y, min_noise_std, max_noise_std, save_dir_train_x, save_dir_train_y)
    test_x, test_y = get_hided_subset_im_paths(test_dirs)
    print('\n\nProcessing test data')
    process_x_y_ims(test_x, test_y, min_noise_std, max_noise_std, save_dir_test_x, save_dir_test_y)


def make_sidd_dset(orig: str,
                   save_dir_train_x: Path, save_dir_train_y: Path,
                   save_dir_test_x: Path, save_dir_test_y: Path,
                   min_noise_std: int, max_noise_std: int) -> None:

    def _process_sidd_test_x_y_ims(noisy_set, gt_set, sdir_x, sdir_y):
        noisy_data, gt_data = loadmat(noisy_set)['ValidationNoisyBlocksSrgb'], loadmat(gt_set)['ValidationGtBlocksSrgb']
        noisy_data = noisy_data.reshape(-1, noisy_data.shape[2], noisy_data.shape[3], noisy_data.shape[4])
        gt_data = gt_data.reshape(-1, gt_data.shape[2], gt_data.shape[3], gt_data.shape[4])
        for x_p, y_p in tqdm(zip(noisy_data, gt_data), total=len(gt_data)):
            im_hash = get_rand_uuid()
            imx_p, imy_p = sdir_x / f'{im_hash}.png', sdir_y / f'{im_hash}.png'
            x_p = cv2.cvtColor(x_p, cv2.COLOR_BGR2RGB)
            y_p = cv2.cvtColor(y_p, cv2.COLOR_BGR2RGB)
            cv2.imwrite(imx_p, x_p)
            cv2.imwrite(imy_p, y_p)

    def _get_test_x_y(test_dir):
        test_files = list(test_dir.glob('*.mat'))
        x_set = [mfile for mfile in test_files if 'Noisy' in mfile.stem][0]
        y_set = [mfile for mfile in test_files if 'Gt' in mfile.stem][0]

        return x_set, y_set

    train_dirs = Path(fr'{orig}/SIDD_Medium_Srgb/Data')
    test_dirs = Path(fr'{orig}/validation')
    # Process train
    train_x, train_y = get_sidd_subset_im_paths(train_dirs)
    # print('\n\nProcessing train data')
    process_x_y_ims(train_x, train_y, min_noise_std, max_noise_std, save_dir_train_x, save_dir_train_y)
    test_x, test_y = _get_test_x_y(test_dirs)
    print('\n\nProcessing test data')
    _process_sidd_test_x_y_ims(test_x, test_y, save_dir_test_x, save_dir_test_y)


def make_renoir_dset(orig: str,
                     save_dir_train: Path,
                     save_dir_test: Path) -> None:

    def _get_train_test_im_paths(orig_p):
        base = Path(orig_p)
        all_ims = base.rglob('*.bmp')
        references = np.array([im for im in all_ims if 'Reference' in str(im)])
        train_set = np.random.choice(references, 100, replace=False)
        test_set = np.random.choice(references[~np.isin(references, train_set)], 20, replace=False)
        return train_set, test_set

    def _extract_patches(image, patch_size, overlap_ratio=0.25):
        step = int(patch_size * (1 - overlap_ratio))
        height, width = image.shape[:2]
        patches = []

        for y in range(0, height - patch_size + 1, step):
            for x in range(0, width - patch_size + 1, step):
                patch = image[y:y + patch_size, x:x + patch_size]
                patches.append(patch)

        return patches

    def _write_set(set_ims, save_set_dir):
        for img in tqdm(set_ims):
            img_arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
            patches = _extract_patches(img_arr, 256)
            for idx, patch in enumerate(patches):
                imsource = str(train_s[-1]).split('\\')[-3]
                batch = str(train_s[-1]).split('\\')[-2]
                im_hash = get_rand_uuid()
                im_name = save_set_dir / f'{imsource}_{batch}_{idx}_{im_hash}.png'
                cv2.imwrite(str(im_name), patch)

    train_s, test_s = _get_train_test_im_paths(orig)
    _write_set(train_s, save_dir_train)
    _write_set(test_s, save_dir_test)


def main():
    args_parser = argparse.ArgumentParser(description='Script to generate dataset with noise and blur')
    args_parser.add_argument('--dset', '-d', type=str, help='Dataset',
                             default=r'renoir')
    args_parser.add_argument('--orig', '-o', type=str, help='Path to orig dset dir',
                             default=r'D:/Projects/datasets/RENOIR')
    args_parser.add_argument('--save_dir', '-s', type=str, help='Dir (relative to cwd) to save images',
                             default=r'D:/Projects/datasets/RENOIR/orig')
    args_parser.add_argument('--min_noise_std', '-m', type=int, help='Minimum std of noise level',
                             default=15)
    args_parser.add_argument('--max_noise_std', '-M', type=int, help='Maximum std of noise level',
                             default=60)
    args = args_parser.parse_args()

    save_dir_train = Path(args.save_dir) / f'awgn-{args.min_noise_std}-{args.max_noise_std}' / 'train'
    save_dir_train_x, save_dir_train_y = save_dir_train / 'x', save_dir_train / 'y'
    save_dir_train_x.mkdir(parents=True, exist_ok=True)
    save_dir_train_y.mkdir(parents=True, exist_ok=True)

    save_dir_test = Path(args.save_dir) / f'awgn-{args.min_noise_std}-{args.max_noise_std}' / 'test'
    save_dir_test_x, save_dir_test_y = save_dir_test / 'x', save_dir_test / 'y'
    save_dir_test_x.mkdir(parents=True, exist_ok=True)
    save_dir_test_y.mkdir(parents=True, exist_ok=True)

    if args.dset == Dset.GOPRO.value:
        make_gopro_dset(args.orig, save_dir_train_x, save_dir_train_y, save_dir_test_x, save_dir_test_y,
                        args.min_noise_std, args.max_noise_std)
    elif args.dset == Dset.HIDE.value:
        make_hide_dset(args.orig, save_dir_train_x, save_dir_train_y, save_dir_test_x, save_dir_test_y,
                       args.min_noise_std, args.max_noise_std)
    elif args.dset == Dset.REALBLUR.value:
        make_realblur_dset(args.orig, save_dir_train_x, save_dir_train_y, save_dir_test_x, save_dir_test_y,
                           args.min_noise_std, args.max_noise_std)
    elif args.dset == Dset.SIDD.value:
        make_sidd_dset(args.orig, save_dir_train_x, save_dir_train_y, save_dir_test_x, save_dir_test_y,
                       args.min_noise_std, args.max_noise_std)
    elif args.dset == Dset.RENOIR.value:
        make_renoir_dset(args.orig, save_dir_train_y, save_dir_test_y)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()