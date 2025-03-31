import os

ROOT_DIR = "./images"
DB_PATH = "./Database/structured_features_ver2.db"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
JSON_OUTPUT = "./inspection/image_features_sample_ver2.json"

CACHE_DIR = "cache"
TREES_FILE = os.path.join(CACHE_DIR, "balltrees.pkl")
ARRAYS_FILE = os.path.join(CACHE_DIR, "arrays.pkl")
PATHS_FILE = os.path.join(CACHE_DIR, "image_paths.pkl")
META_FILE = os.path.join(CACHE_DIR, "meta.pkl")