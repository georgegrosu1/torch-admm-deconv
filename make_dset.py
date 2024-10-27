import argparse
from pathlib import Path


def main():
    args_parser = argparse.ArgumentParser(description='Script to generate dataset with noise and blur')
    args_parser.add_argument('--orig', '-o', type=str, help='Path to dir with original images',
                             default=r'configs/train_cfg.json')
    args_parser.add_argument('--save_dir', '-s', type=str, help='Dir (relative to cwd) to save images',
                             default=r'real_blur_modif')
    args_parser.add_argument('--min_noise_std', '-m', type=int, help='Minimum std of noise level',
                             default=15)
    args_parser.add_argument('--max_noise_std', '-M', type=int, help='Maximum std of noise level',
                             default=60)
    args = args_parser.parse_args()


if __name__ == "__main__":
    main()
