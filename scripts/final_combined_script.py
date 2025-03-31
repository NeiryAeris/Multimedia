import os
import gc
import json
import sqlite3
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

#-------------------Search served packages#-------------------

from sklearn.metrics.pairwise import cosine_similarity

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
    hsv = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h_hist, s_hist, v_hist])
    return (hist / hist.sum()).tolist()

def extract_shape_descriptor(img):
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return np.log1p(np.abs(hu)).tolist()

def extract_texture_descriptor(img): # could use Gabor filters, Haralick features, or GLCM if want more robust texture features
    gray = rgb2gray(img.astype("uint8"))
    lbp = local_binary_pattern(gray, P=8, R=1.0)
    hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
    return (hist / hist.sum()).tolist()

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


# ----- Below is the original notebook content -----

#!/usr/bin/env python
# coding: utf-8

# ### Packages importing
# #### this action should be done 1 time only cus it took pretty long to process




# In[25]:


ROOT_DIR = "images"
# DB_PATH = "./Database/structured_features.db"
DB_PATH = "./Database/structured_features_ver2.db"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
# JSON_OUTPUT = "./inspection/image_features_sample.json"
JSON_OUTPUT = "./inspection/image_features_sample_ver2.json"


# **Enviroment** config loading and feature **model** loading funcion

# In[26]:


def configure_environment():
    mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

def load_feature_model():
    return VGG16(weights="imagenet", include_top=False, pooling="avg")


# Database init and features storing queries

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS image_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            label TEXT,
            color_histogram TEXT,
            shape_descriptor TEXT,
            texture_descriptor TEXT,
            deep_embedding TEXT
        )
    """)
    return conn, c

def insert_features(cursor, data):
    cursor.executemany("""
        INSERT INTO image_features (
            image_path, label, color_histogram, shape_descriptor, texture_descriptor, deep_embedding
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, data)

def save_sample(cursor):
    cursor.execute("SELECT * FROM image_features LIMIT 10;")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    with open(JSON_OUTPUT, "w") as f:
        json.dump([dict(zip(columns, row)) for row in rows], f, indent=4)


def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    return img_to_array(img)

def preprocess_batch(images):
    return preprocess_input(np.array(images)).astype("float16")

def extract_features():
    configure_environment()
    model = load_feature_model()
    conn, c = init_db()

    image_batch, path_batch, label_batch, raw_images = [], [], [], []

    for class_dir in os.listdir(ROOT_DIR):
        class_path = os.path.join(ROOT_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            img = load_image(image_path)
            raw_images.append(img)
            image_batch.append(img)
            path_batch.append(image_path)
            label_batch.append(class_dir)

            if len(image_batch) == BATCH_SIZE:
                process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, c)
                image_batch, path_batch, label_batch, raw_images = [], [], [], []

    if image_batch:
        process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, c)

    save_sample(c)
    conn.commit()
    conn.close()

def process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, cursor):
    batch_np = preprocess_batch(image_batch)
    features = model.predict(batch_np, verbose=0)

    insert_data = []

    for i in range(len(features)):
        img_raw = raw_images[i]
        color_hist = extract_color_histogram(img_raw)
        shape_desc = extract_shape_descriptor(img_raw)
        texture_desc = extract_texture_descriptor(img_raw)
        deep_embed = features[i].flatten().tolist()

        insert_data.append((
            path_batch[i],
            label_batch[i],
            json.dumps(color_hist),
            json.dumps(shape_desc),
            json.dumps(texture_desc),
            json.dumps(deep_embed)
        ))

    insert_features(cursor, insert_data)
    K.clear_session()
    gc.collect()



if __name__ == "__main__":
    extract_features()
