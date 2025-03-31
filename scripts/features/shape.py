import cv2
import numpy as np

def extract_shape_descriptor(img):
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return np.log1p(np.abs(hu)).tolist()
