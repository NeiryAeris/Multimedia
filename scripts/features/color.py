import numpy as np
import cv2

def extract_color_histogram(img, bins=32):
    hsv = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h_hist, s_hist, v_hist])
    return (hist / hist.sum()).tolist()
