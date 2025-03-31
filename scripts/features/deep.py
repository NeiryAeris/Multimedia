import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from model import load_feature_model

def preprocess_batch(images):
    return preprocess_input(np.array(images)).astype("float16")

def extract_deep_embedding(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return load_feature_model().predict(x, verbose=0).flatten().tolist()