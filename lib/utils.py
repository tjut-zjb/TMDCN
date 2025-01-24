import glob
import logging
import os
import re
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def getLogger(save_dir):
    log_dir = os.path.join(save_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.txt")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filename,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True


def increment_path(save_dir, sep='-'):
    """
    Increment path, i.e. run/train/exp --> run/train/exp{sep}2, run/train/exp{sep}3 etc.
    """
    path = Path(save_dir)
    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 1  # increment number
    return f"{path}{sep}{n}"  # update path


def getSaveDir(save_path, save_name, continue_training, continue_training_name=None):
    if continue_training:
        if continue_training_name:
            save_dir = os.path.join(save_path, continue_training_name)
            if not os.path.exists(save_dir):
                raise FileNotFoundError(f"Specified training directory '{continue_training_name}' not found.")
        else:
            raise ValueError("continue_training is True but no continue_training_name is specified.")
    else:
        save_dir = increment_path(Path(save_path) / save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(epoch, models, optimizer, best_val_loss,
                    val_avg_loss, val_avg_mae, val_avg_mape, val_avg_rmse,
                    save_dir, name):
    checkpoint = {
        'epoch': epoch,
        'models_state_dict': {name: model.state_dict() for name, model in models.items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }

    checkpoint_save_dir = os.path.join(save_dir, 'checkpoint')

    if not os.path.exists(checkpoint_save_dir):
        os.makedirs(checkpoint_save_dir)

    torch.save(checkpoint, os.path.join(checkpoint_save_dir, f'{name}.pt'))

    with open(os.path.join(checkpoint_save_dir, f'{name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Epoch: {epoch + 1}\n"
                f"Loss: {val_avg_loss:.2f}\n"
                f"MAE: {val_avg_mae:.2f}\n"
                f"MAPE: {val_avg_mape:.2f}\n"
                f"RMSE: {val_avg_rmse:.2f}\n")
