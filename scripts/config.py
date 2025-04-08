import os

ROOT_DIR = "./Raw"
# DB_PATH = "./Database/structured_features_ver2.db"
DB_PATH = "./Database/structured_features_ver3.db"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
# JSON_OUTPUT = "./inspection/image_features_sample_ver2.json"
JSON_OUTPUT = "./inspection/image_features_sample_ver3.json"

CACHE_DIR = "cache"
# TREES_FILE = os.path.join(CACHE_DIR, "balltrees.pkl")
# ARRAYS_FILE = os.path.join(CACHE_DIR, "arrays.pkl")
# PATHS_FILE = os.path.join(CACHE_DIR, "image_paths.pkl")
# META_FILE = os.path.join(CACHE_DIR, "meta.pkl")

TREES_FILE = os.path.join(CACHE_DIR, "balltrees_ver2.pkl")
ARRAYS_FILE = os.path.join(CACHE_DIR, "arrays_ver2.pkl")
PATHS_FILE = os.path.join(CACHE_DIR, "image_paths_ver2.pkl")
META_FILE = os.path.join(CACHE_DIR, "meta_ver2.pkl")