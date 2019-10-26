import numpy as np
from keras.preprocessing import image


def merge_labels_and_vals_to_dict(labels, values):
    return dict(zip(labels, values))

def load_input_img(filepath, shape):
    img = image.load_img(filepath, target_size=shape)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)