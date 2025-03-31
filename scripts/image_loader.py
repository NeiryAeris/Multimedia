from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import IMG_SIZE

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    return img_to_array(img)
