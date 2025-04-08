import json
import sqlite3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_loader import load_image
from features.color import extract_color_histogram
from features.shape import extract_shape_descriptor
from features.texture import extract_texture_descriptor
from model import load_feature_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from utils import compute_cosine_sim

from balltree_cache import load_or_build_balltrees

trees, arrays, image_paths = load_or_build_balltrees()

model = load_feature_model()

def extract_deep_embedding(img):
    batch = preprocess_input(img.reshape((1, 224, 224, 3)))
    return model.predict(batch)[0].tolist()

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

        total_similarity = (
            0.15 * sims["color"] +
            0.2 * sims["shape"] +
            0.2 * sims["texture"] +
            0.45 * sims["deep"]
        )

        results.append((image_path, total_similarity, sims))

    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
