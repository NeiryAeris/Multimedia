import cv2
import numpy as np

def extract_shape_descriptor(img):
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return np.log1p(np.abs(hu)).tolist()

def extract_hog_hu(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    hog_features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(1, 1),
        visualize=True,
        block_norm='L2'
    )

    # Extract Hu Moments
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = np.log1p(np.abs(hu_moments))  # Stability for large values

    # Combine features
    combined = np.concatenate([hog_features, hu_moments])
    return combined