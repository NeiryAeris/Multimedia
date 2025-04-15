import numpy as np
import cv2
from skimage.feature import hog

def extract_color_histogram(img, bins=32):
    hsv = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h_hist, s_hist, v_hist])
    return (hist / hist.sum()).tolist()

def extract_hog(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (fd, hog_image) = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L2')
    return fd

def extract_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_rgb, hist_rgb)
    return hist_rgb.flatten()

def extract_hog_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L2')
    cv2.normalize(hist_rgb, hist_rgb)
    combined_features = np.concatenate((fd, hist_rgb.flatten()))
    return combined_features