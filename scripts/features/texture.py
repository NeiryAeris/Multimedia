import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

def extract_texture_descriptor(img):
    gray = rgb2gray(img.astype("uint8"))
    lbp = local_binary_pattern(gray, P=8, R=1.0)
    hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
    return (hist / hist.sum()).tolist()
