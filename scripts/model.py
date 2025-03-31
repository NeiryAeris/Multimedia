import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16

def configure_environment():
    mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

def load_feature_model():
    return VGG16(weights="imagenet", include_top=False, pooling="avg")
