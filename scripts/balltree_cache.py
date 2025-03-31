
import os
import sqlite3
import json
import numpy as np
import pickle
from sklearn.neighbors import BallTree

from config import DB_PATH, CACHE_DIR ,TREES_FILE, ARRAYS_FILE, PATHS_FILE, META_FILE
from database import load_database_features

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_db_meta():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MAX(id) FROM image_features")
    row_count, max_id = cursor.fetchone()
    conn.close()
    return {"row_count": row_count, "max_id": max_id}

def is_cache_valid():
    if not os.path.exists(META_FILE):
        return False
    with open(META_FILE, "rb") as f:
        cached_meta = pickle.load(f)
    return cached_meta == get_db_meta()

def load_cache():
    with open(TREES_FILE, "rb") as f:
        trees = pickle.load(f)
    with open(ARRAYS_FILE, "rb") as f:
        arrays = pickle.load(f)
    with open(PATHS_FILE, "rb") as f:
        image_paths = pickle.load(f)
    return trees, arrays, image_paths

def save_cache(trees, arrays, image_paths):
    with open(TREES_FILE, "wb") as f:
        pickle.dump(trees, f)
    with open(ARRAYS_FILE, "wb") as f:
        pickle.dump(arrays, f)
    with open(PATHS_FILE, "wb") as f:
        pickle.dump(image_paths, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(get_db_meta(), f)

# def load_database_features():
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT image_path, color_histogram, shape_descriptor, texture_descriptor, deep_embedding FROM image_features")
#     rows = cursor.fetchall()
#     conn.close()

#     image_paths = []
#     color_features = []
#     shape_features = []
#     texture_features = []
#     deep_features = []

#     for row in rows:
#         image_paths.append(row[0])
#         color_features.append(json.loads(row[1]))
#         shape_features.append(json.loads(row[2]))
#         texture_features.append(json.loads(row[3]))
#         deep_features.append(json.loads(row[4]))

#     return image_paths, np.array(color_features), np.array(shape_features), np.array(texture_features), np.array(deep_features)

def build_balltrees(color_array, shape_array, texture_array, deep_array):
    return {
        "color": BallTree(color_array, metric='euclidean'),
        "shape": BallTree(shape_array, metric='euclidean'),
        "texture": BallTree(texture_array, metric='euclidean'),
        "deep": BallTree(deep_array, metric='euclidean')
    }

def load_or_build_balltrees():
    ensure_cache_dir()
    if is_cache_valid():
        print("✅ Loaded BallTrees from cache.")
        return load_cache()
    print("⚙️ Rebuilding BallTrees from database...")
    image_paths, color_array, shape_array, texture_array, deep_array = load_database_features()
    arrays = (color_array, shape_array, texture_array, deep_array)
    trees = build_balltrees(color_array, shape_array, texture_array, deep_array)
    save_cache(trees, arrays, image_paths)
    return trees, arrays, image_paths
