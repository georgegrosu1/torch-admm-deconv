import argparse
from tqdm import tqdm
from pathlib import Path

from utils.dset_utils import *


def get_train_test_txts(orig_p: Path) -> Tuple[List, List]:
    train_txts = list(orig_p.glob('*train.txt'))
    test_txts = list(orig_p.glob('*test.txt'))
    return train_txts, test_txts


def process_x_y_ims(x_ims: List, y_ims: List, min_stdv, max_stdv, sdir_x, sdir_y) -> None:
    for x_p, y_p in tqdm(zip(x_ims, y_ims), total=len(y_ims)):
        im_hash = get_rand_uuid()
        imx, imy = cv2.imread(str(x_p), cv2.IMREAD_COLOR), cv2.imread(str(y_p), cv2.IMREAD_COLOR)
        if max_stdv > min_stdv >= 0:
            r_stdv = np.random.randint(min_stdv, max_stdv)
            imx = add_noise_gaussian(imx, stdv=r_stdv)
        else:
            r_stdv = 0
        imx_p, imy_p = sdir_x / f'{im_hash}_awgn-{r_stdv}.png', sdir_y / f'{im_hash}.png'
        cv2.imwrite(imx_p, imx)
        cv2.imwrite(imy_p, imy)


def main():
    args_parser = argparse.ArgumentParser(description='Script to generate dataset with noise and blur')
    args_parser.add_argument('--orig', '-o', type=str, help='Path to RealBlur dir',
                             default=r'D:/Projects/datasets/RealBlur')
    args_parser.add_argument('--save_dir', '-s', type=str, help='Dir (relative to cwd) to save images',
                             default=r'D:/Projects/datasets/RealBlur/orig_blur')
    args_parser.add_argument('--min_noise_std', '-m', type=int, help='Minimum std of noise level',
                             default=15)
    args_parser.add_argument('--max_noise_std', '-M', type=int, help='Maximum std of noise level',
                             default=60)
    args = args_parser.parse_args()

    train_txts, test_txts = get_train_test_txts(Path(args.orig))

    save_dir_train = Path(args.save_dir) / f'awgn-{args.min_noise_std}-{args.max_noise_std}' / 'train'
    save_dir_train_x, save_dir_train_y = save_dir_train / 'x', save_dir_train / 'y'
    save_dir_train_x.mkdir(parents=True, exist_ok=True)
    save_dir_train_y.mkdir(parents=True, exist_ok=True)

    save_dir_test = Path(args.save_dir) / f'awgn-{args.min_noise_std}-{args.max_noise_std}' / 'test'
    save_dir_test_x, save_dir_test_y = save_dir_test / 'x', save_dir_test / 'y'
    save_dir_test_x.mkdir(parents=True, exist_ok=True)
    save_dir_test_y.mkdir(parents=True, exist_ok=True)

    # Process train
    train_1_x, train_1_y = get_dset_im_paths(train_txts[0])
    train_2_x, train_2_y = get_dset_im_paths(test_txts[1])
    train_x, train_y = train_1_x + train_2_x, train_1_y + train_2_y
    print('\n\nProcessing train data')
    process_x_y_ims(train_x, train_y, args.min_noise_std, args.max_noise_std, save_dir_train_x, save_dir_train_y)

    # Process test
    test_1_x, test_1_y = get_dset_im_paths(test_txts[0])
    test_2_x, test_2_y = get_dset_im_paths(test_txts[1])
    test_x, test_y = test_1_x + test_2_x, test_1_y + test_2_y
    print('\n\nProcessing test data')
    process_x_y_ims(test_x, test_y, args.min_noise_std, args.max_noise_std, save_dir_test_x, save_dir_test_y)


if __name__ == "__main__":
    main()
