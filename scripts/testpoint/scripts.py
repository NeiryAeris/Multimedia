import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from features.color import extract_color_histogram, extract_hog, extract_rgb, extract_hog_rgb
from image_loader import load_image

raw_images = []

image_path = './sample_query.jpg'
img = load_image(image_path)
raw_images.append(img)

vec1 = extract_color_histogram(img)
vec2 = extract_hog(image_path)
vec3 = extract_rgb(image_path)
vec4 = extract_hog_rgb(image_path)

print("Color Histogram:", vec1)
print("HOG:", vec2)
print("RGB Histogram:", vec3)
print("HOG + RGB:", vec4)
print("Length of Color Histogram:", len(vec1))
print("Length of HOG:", len(vec2))
print("Length of RGB Histogram:", len(vec3))
print("Length of HOG + RGB:", len(vec4))