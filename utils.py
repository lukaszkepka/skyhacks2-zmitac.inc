import numpy as np
from keras.preprocessing import image
import pandas as pd


def merge_labels_and_vals_to_dict(labels, values):
    return dict(zip(labels, values))


def load_input_img(filepath, shape):
    img = image.load_img(filepath, target_size=shape)
    x = image.img_to_array(img) / 255
    return np.expand_dims(x, axis=0)


def predict_rooms_labels(y_pred_proba, rooms_labels, threshold=0.3):
    rooms = [list(np.where(sample > threshold)[0]) for sample in y_pred_proba]
    rooms_labels = [[(1 if idx in rooms[i] else 0) for idx in range(len(rooms_labels))] for i in range(len(rooms))]
    rooms_labels = pd.DataFrame.from_records(rooms_labels)
    rooms_labels.columns = rooms_labels
    return rooms_labels