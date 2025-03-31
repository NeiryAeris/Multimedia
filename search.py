import os
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sqlite3
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import mixed_precision

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Load VGG16 without top layer and with global average pooling
model = VGG16(weights="imagenet", include_top=False, pooling="avg")

# Enable mixed precision AFTER model definition
mixed_precision.set_global_policy("mixed_float16")

# Function to extract feature vector from an image
def extract_features_for_query(query_img_path):
    img = load_img(query_img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array).astype("float16")  # Match mixed precision
    return model.predict(img_array, verbose=0).flatten()

# Function to search similar images using cosine similarity
def search_image(query_img_path, top_n=5):
    query_vector = extract_features_for_query(query_img_path).reshape(1, -1)

    conn = sqlite3.connect("features.db")
    c = conn.cursor()
    c.execute("SELECT image_path, features FROM image_features")

    results = []
    for row in c.fetchall():
        path, features_str = row
        feature_vec = np.array(list(map(float, features_str.split(",")))).reshape(1, -1)
        similarity = cosine_similarity(query_vector, feature_vec)[0][0]
        results.append((path, similarity))

    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# Run image search
results = search_image("sample_query.jpg", top_n=3)
for path, sim in results:
    print(f"{path} - Similarity: {sim:.4f}")
