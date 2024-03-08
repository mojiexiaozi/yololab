# yololab YOLO 🚀, AGPL-3.0 license
"""Monkey patches to update/extend functionality of existing functions."""

import time
from pathlib import Path

import cv2
import numpy as np
import torch

# OpenCV Multilanguage-friendly functions ------------------------------------------------------------------------------
_imshow = cv2.imshow  # copy to avoid recursion errors


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch functions ----------------------------------------------------------------------------------------------------
_torch_save = torch.save  # copy to avoid recursion errors


def torch_save(*args, use_dill=True, **kwargs):
    try:
        assert use_dill
        import dill as pickle
    except (AssertionError, ImportError):
        import pickle

    if "pickle_module" not in kwargs:
        kwargs["pickle_module"] = pickle

    for i in range(4):  # 3 retries
        try:
            return _torch_save(*args, **kwargs)
        except (
            RuntimeError
        ) as e:  # unable to save, possibly waiting for device to flush or antivirus scan
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # exponential standoff: 0.5s, 1.0s, 2.0s
