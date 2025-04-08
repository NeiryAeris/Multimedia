
import os
import json
import numpy as np
import sqlite3
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

# ------------------ Load VGG Model ------------------
model = VGG16(weights="imagenet", include_top=False, pooling="avg")

# ------------------ Feature Extraction ------------------

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
    x = vgg16.preprocess_input(x)
    return model.predict(x, verbose=0).flatten().tolist()

# ------------------ Load Database Features ------------------

def load_database_features(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, color_histogram, shape_descriptor, texture_descriptor, deep_embedding FROM image_features")
    rows = cursor.fetchall()
    conn.close()

    image_paths = []
    color_features = []
    shape_features = []
    texture_features = []
    deep_features = []

    for row in rows:
        image_paths.append(row[0])
        color_features.append(json.loads(row[1]))
        shape_features.append(json.loads(row[2]))
        texture_features.append(json.loads(row[3]))
        deep_features.append(json.loads(row[4]))

    return image_paths, np.array(color_features), np.array(shape_features), np.array(texture_features), np.array(deep_features)

# ------------------ Optimized Search ------------------

def build_balltrees(color_array, shape_array, texture_array, deep_array):
    return {
        "color": BallTree(color_array, metric='euclidean'),
        "shape": BallTree(shape_array, metric='euclidean'),
        "texture": BallTree(texture_array, metric='euclidean'),
        "deep": BallTree(deep_array, metric='euclidean')
    }

def search_image_optimized(query_features, trees, arrays, image_paths, top_k_tree=30, top_n_final=5):
    q_color = np.array(query_features["color_histogram"]).reshape(1, -1)
    q_shape = np.array(query_features["shape_descriptor"]).reshape(1, -1)
    q_texture = np.array(query_features["texture_descriptor"]).reshape(1, -1)
    q_deep = np.array(query_features["deep_embedding"]).reshape(1, -1)

    deep_tree = trees["deep"]
    dist, indices = deep_tree.query(q_deep, k=top_k_tree)

    color_array, shape_array, texture_array, deep_array = arrays

    results = []
    for idx in indices[0]:
        db_color = color_array[idx]
        db_shape = shape_array[idx]
        db_texture = texture_array[idx]
        db_deep = deep_array[idx]

        sims = {
            "color": cosine_similarity(q_color, db_color.reshape(1, -1))[0][0],
            "shape": cosine_similarity(q_shape, db_shape.reshape(1, -1))[0][0],
            "texture": cosine_similarity(q_texture, db_texture.reshape(1, -1))[0][0],
            "deep": cosine_similarity(q_deep, db_deep.reshape(1, -1))[0][0],
        }

        total_similarity = (
            0.2 * sims["color"] +
            0.2 * sims["shape"] +
            0.2 * sims["texture"] +
            0.4 * sims["deep"]
        )

        results.append((image_paths[idx], total_similarity, sims))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n_final]
