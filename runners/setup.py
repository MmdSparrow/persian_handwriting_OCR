import os
import random
from pathlib import Path

import mne
import numpy as np
import pyvista as pv
import torch

from utils.setup_utils import check_mne_fsaverage_data_is_presented

__all__ = [
    "set_seed",
    "setup_mne_3d_visualization",
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def setup_mne_3d_visualization(running_on_colab):
    subjects_dir = None
    if check_mne_fsaverage_data_is_presented():
        print("MNE fsaverage data already exist.")
        root_path = Path.home()
        subjects_dir = root_path / "mne_data" / "MNE-fsaverage-data"
    else:
        print("MNE fsaverage data not exist. try to download...")
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = fs_dir.parent

    # os.environ["SUBJECTS_DIR"] = subjects_dir
    mne.set_config(
        "SUBJECTS_DIR",
        str(subjects_dir),
        set_env=True,
    )

    if running_on_colab:
        print("running on colab...")
        pv.OFF_SCREEN = True
        # mne.viz.set_3d_backend("pyvista")
        mne.viz.set_3d_backend("notebook")
    else:
        print("running on local...")
        mne.viz.set_3d_backend("pyvistaqt")
