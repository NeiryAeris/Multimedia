import os
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
import gc
import json

# Enable mixed precision for performance
mixed_precision.set_global_policy("mixed_float16")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Load VGG16 with Global Average Pooling
model = VGG16(weights="imagenet", include_top=False, pooling="avg")

# Connect to SQLite and create table
conn = sqlite3.connect("features.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS image_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        label TEXT,
        features TEXT
    )
""")

# Batch feature extraction
root_dir = "images"
batch_size = 32
image_batch, path_batch, label_batch = [], [], []

for class_dir in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_dir)
    if not os.path.isdir(class_path):
        continue

    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        image_batch.append(img_array)
        path_batch.append(image_path)
        label_batch.append(class_dir)

        if len(image_batch) == batch_size:
            batch_np = preprocess_input(np.array(image_batch)).astype("float16")
            features = model.predict(batch_np, verbose=0)

            insert_data = [
                (path_batch[i], label_batch[i], ",".join(map(str, features[i].flatten())))
                for i in range(len(features))
            ]
            c.executemany("INSERT INTO image_features (image_path, label, features) VALUES (?, ?, ?)", insert_data)

            image_batch, path_batch, label_batch = [], [], []
            K.clear_session()
            gc.collect()

# Process leftover images
if image_batch:
    batch_np = preprocess_input(np.array(image_batch)).astype("float16")
    features = model.predict(batch_np, verbose=0)

    insert_data = [
        (path_batch[i], label_batch[i], ",".join(map(str, features[i].flatten())))
        for i in range(len(features))
    ]
    c.executemany("INSERT INTO image_features (image_path, label, features) VALUES (?, ?, ?)", insert_data)

# Save first 10 entries to JSON for inspection
c.execute("SELECT * FROM image_features LIMIT 10;")
rows = c.fetchall()
columns = [desc[0] for desc in c.description]
data = [dict(zip(columns, row)) for row in rows]

with open("image_features_sample.json", "w") as f:
    json.dump(data, f, indent=4)

conn.commit()
conn.close()
