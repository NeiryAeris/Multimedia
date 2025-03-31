# import os
# import gc
# import json
# import sqlite3
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras import mixed_precision
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from skimage.color import rgb2gray
# from skimage.feature import local_binary_pattern

# # ------------------ CONFIG ------------------

# ROOT_DIR = "images"
# DB_PATH = "features.db"
# BATCH_SIZE = 32
# IMG_SIZE = (224, 224)
# JSON_OUTPUT = "image_features_sample.json"

# # ------------------ ENV SETUP ------------------

# def configure_environment():
#     mixed_precision.set_global_policy("mixed_float16")
#     gpus = tf.config.experimental.list_physical_devices("GPU")
#     if gpus:
#         try:
#             tf.config.experimental.set_memory_growth(gpus[0], True)
#         except RuntimeError as e:
#             print(e)

# def load_feature_model():
#     return VGG16(weights="imagenet", include_top=False, pooling="avg")

# # ------------------ DB UTILS ------------------

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS image_features (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             image_path TEXT,
#             label TEXT,
#             color_histogram TEXT,
#             shape_descriptor TEXT,
#             texture_descriptor TEXT,
#             deep_embedding TEXT
#         )
#     """)
#     return conn, c

# def insert_features(cursor, data):
#     cursor.executemany("""
#         INSERT INTO image_features (
#             image_path, label, color_histogram, shape_descriptor, texture_descriptor, deep_embedding
#         ) VALUES (?, ?, ?, ?, ?, ?)
#     """, data)

# def save_sample(cursor):
#     cursor.execute("SELECT * FROM image_features LIMIT 10;")
#     rows = cursor.fetchall()
#     columns = [desc[0] for desc in cursor.description]
#     with open(JSON_OUTPUT, "w") as f:
#         json.dump([dict(zip(columns, row)) for row in rows], f, indent=4)

# # ------------------ IMAGE UTILS ------------------

# def load_image(path):
#     img = load_img(path, target_size=IMG_SIZE)
#     return img_to_array(img)

# def preprocess_batch(images):
#     return preprocess_input(np.array(images)).astype("float16")

# # ------------------ FEATURE EXTRACTORS ------------------

# def extract_color_histogram(img, bins=32):
#     hist = np.histogram(img, bins=bins, range=(0, 256))[0]
#     return (hist / hist.sum()).tolist()

# def extract_shape_descriptor(img):
#     return (img.mean(axis=2).flatten()[:32] / 255.0).tolist()

# def extract_texture_descriptor(img):
#     gray = rgb2gray(img.astype("uint8"))
#     lbp = local_binary_pattern(gray, P=8, R=1.0)
#     hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
#     return (hist / hist.sum()).tolist()

# # ------------------ MAIN EXTRACTION ------------------

# def extract_features():
#     configure_environment()
#     model = load_feature_model()
#     conn, c = init_db()

#     image_batch, path_batch, label_batch, raw_images = [], [], [], []

#     for class_dir in os.listdir(ROOT_DIR):
#         class_path = os.path.join(ROOT_DIR, class_dir)
#         if not os.path.isdir(class_path):
#             continue

#         for image_file in os.listdir(class_path):
#             image_path = os.path.join(class_path, image_file)
#             img = load_image(image_path)
#             raw_images.append(img)
#             image_batch.append(img)
#             path_batch.append(image_path)
#             label_batch.append(class_dir)

#             if len(image_batch) == BATCH_SIZE:
#                 process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, c)
#                 image_batch, path_batch, label_batch, raw_images = [], [], [], []

#     if image_batch:
#         process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, c)

#     save_sample(c)
#     conn.commit()
#     conn.close()

# def process_and_store_batch(model, image_batch, raw_images, path_batch, label_batch, cursor):
#     batch_np = preprocess_batch(image_batch)
#     features = model.predict(batch_np, verbose=0)

#     insert_data = []

#     for i in range(len(features)):
#         img_raw = raw_images[i]
#         color_hist = extract_color_histogram(img_raw)
#         shape_desc = extract_shape_descriptor(img_raw)
#         texture_desc = extract_texture_descriptor(img_raw)
#         deep_embed = features[i].flatten().tolist()

#         insert_data.append((
#             path_batch[i],
#             label_batch[i],
#             json.dumps(color_hist),
#             json.dumps(shape_desc),
#             json.dumps(texture_desc),
#             json.dumps(deep_embed)
#         ))

#     insert_features(cursor, insert_data)
#     K.clear_session()
#     gc.collect()

# # ------------------ RUN ------------------

# if __name__ == "__main__":
#     extract_features()

