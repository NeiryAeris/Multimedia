import cv2
import os
import matplotlib.pyplot as plt
from feature_extraction import orb

def visualize_features(image_path, output_path="results/"):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, _ = orb.detectAndCompute(gray, None)
    
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
    
    os.makedirs(output_path, exist_ok=True)
    result_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(result_path, img_with_keypoints)

    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
