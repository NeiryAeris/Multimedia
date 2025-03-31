import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree
from database import load_database_features
import os
import pickle
import sqlite3

from config import DB_PATH, CACHE_DIR ,TREES_FILE, ARRAYS_FILE, PATHS_FILE, META_FILE

def compute_cosine_sim(vec1, vec2):
    return cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]

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