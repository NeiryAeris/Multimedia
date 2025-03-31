from model import configure_environment, load_feature_model
from scripts.image_loader import load_image
from features.color import extract_color_histogram
from features.shape import extract_shape_descriptor
from features.texture import extract_texture_descriptor
from features.deep import preprocess_batch
from database import init_db, insert_features, save_sample
from tensorflow.keras import backend as K
import gc
import os
from config import ROOT_DIR, BATCH_SIZE
import json

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

if __name__ == "__main__":
    extract_features()
