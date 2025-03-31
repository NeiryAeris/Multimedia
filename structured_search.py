import os
import numpy as np
import sqlite3
import tensorflow as tf
import json
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import mixed_precision
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

# ------------------ SETUP ------------------

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Load model
model = VGG16(weights="imagenet", include_top=False, pooling="avg")
mixed_precision.set_global_policy("mixed_float16")

# ------------------ FEATURE EXTRACTORS ------------------

def load_image(path):
    img = load_img(path, target_size=(224, 224))
    return img_to_array(img)

def extract_color_histogram(img, bins=32):
    hist = np.histogram(img, bins=bins, range=(0, 256))[0]
    return (hist / hist.sum()).tolist()

def extract_shape_descriptor(img):
    return (img.mean(axis=2).flatten()[:32] / 255.0).tolist()

def extract_texture_descriptor(img):
    gray = rgb2gray(img.astype("uint8"))
    lbp = local_binary_pattern(gray, P=8, R=1.0)
    hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
    return (hist / hist.sum()).tolist()

def extract_deep_embedding(img):
    x = np.expand_dims(img, axis=0)
    x = vgg16.preprocess_input(x).astype("float16")
    return model.predict(x, verbose=0).flatten().tolist()

# ------------------ SIMILARITY ------------------

def compute_cosine_sim(vec1, vec2):
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

# ------------------ SEARCH FUNCTION ------------------

def search_image(query_img_path, top_n=5):
    img = load_image(query_img_path)

    query_features = {
        "color_histogram": extract_color_histogram(img),
        "shape_descriptor": extract_shape_descriptor(img),
        "texture_descriptor": extract_texture_descriptor(img),
        "deep_embedding": extract_deep_embedding(img)
    }

    conn = sqlite3.connect("structured_features.db")
    c = conn.cursor()
    c.execute("SELECT image_path, color_histogram, shape_descriptor, texture_descriptor, deep_embedding FROM image_features")

    results = []

    for row in c.fetchall():
        image_path, color_str, shape_str, texture_str, deep_str = row

        db_features = {
            "color_histogram": json.loads(color_str),
            "shape_descriptor": json.loads(shape_str),
            "texture_descriptor": json.loads(texture_str),
            "deep_embedding": json.loads(deep_str)
        }

        sims = {
            "color": compute_cosine_sim(query_features["color_histogram"], db_features["color_histogram"]),
            "shape": compute_cosine_sim(query_features["shape_descriptor"], db_features["shape_descriptor"]),
            "texture": compute_cosine_sim(query_features["texture_descriptor"], db_features["texture_descriptor"]),
            "deep": compute_cosine_sim(query_features["deep_embedding"], db_features["deep_embedding"]),
        }

        # Weighted average (customizable)
        total_similarity = (
            0.2 * sims["color"] +
            0.2 * sims["shape"] +
            0.2 * sims["texture"] +
            0.4 * sims["deep"]
        )

        results.append((image_path, total_similarity, sims))

    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# ------------------ RUN SEARCH ------------------

results = search_image("sample_query.jpg", top_n=3)
for path, total_sim, sims in results:
    print(f"\nImage: {path}")
    print(f"  ➤ Total Similarity: {total_sim:.4f}")
    print(f"     - Color:   {sims['color']:.4f}")
    print(f"     - Shape:   {sims['shape']:.4f}")
    print(f"     - Texture: {sims['texture']:.4f}")
    print(f"     - Deep:    {sims['deep']:.4f}")
