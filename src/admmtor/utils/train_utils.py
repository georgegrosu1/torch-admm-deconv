import time
from pathlib import Path


def get_abs_path(relative_path) -> Path:
    root_path = Path(__file__).resolve().parent.parent
    return Path(str(root_path) + f'{relative_path}')


def get_saving_model_path(save_path: str, model_name: str, save_time: str = None) -> Path:
    if save_time is not None:
        save_dir = Path(save_path) / model_name / save_time
    else:
        save_dir = Path(save_path) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name + '_epoch{epoch:02d}_vloss{val_loss:.4f}'
    return save_dir / model_name


def get_x_y_paths(x_dir, y_dir):
    x_path = get_abs_path(x_dir)
    y_path = get_abs_path(y_dir)

    return x_path, y_path


def get_time_formated() -> str:
    return time.ctime().replace(':', '_').replace(' ', '-')