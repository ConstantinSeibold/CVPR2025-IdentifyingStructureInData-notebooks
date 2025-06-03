from skimage import color
import numpy as np
from collections import Counter
from scipy.ndimage import label as cc_label
from skimage.util import img_as_float
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def enforce_connectivity(labels: np.ndarray, min_size: int = 20) -> np.ndarray:
    """
    Merge any connected component smaller than min_size into the most
    frequent neighboring label.
    """
    H, W = labels.shape
    new = labels.copy()
    for lab in np.unique(labels):
        mask = (labels == lab)
        comp, n_comp = cc_label(mask, structure=np.ones((3,3),int))
        if n_comp <= 1:
            continue
        counts = Counter(comp.flat)
        counts.pop(0, None)
        main_cc = max(counts, key=counts.get)
        for cc_id, sz in counts.items():
            if cc_id == main_cc or sz > min_size:
                continue
            ys, xs = np.where(comp == cc_id)
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y-1), min(H-1, y+1)
                x0, x1 = max(0, x-1), min(W-1, x+1)
                neigh = new[y0:y1+1, x0:x1+1].flat
                choices = [int(v) for v in neigh if v != lab]
                if choices:
                    new[y, x] = Counter(choices).most_common(1)[0][0]
    return new

def get_pixel_coordinates(H, W):
    """Generate coordinate features for an H×W image."""
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    return np.stack([yy, xx], axis=-1).reshape(-1, 2)  # (H*W, 2)
