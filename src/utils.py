import numpy as np
from keras.preprocessing import image

def merge_labels_and_vals_to_dict(labels, values):
    return dict(zip(labels, values))


def load_input_img(filepath, shape):
    img = image.load_img(filepath, target_size=shape)
    x = image.img_to_array(img) / 255
    return np.expand_dims(x, axis=0)


def predict_rooms_labels(y_pred_proba, rooms_labels, threshold=0.3):
    assert len(y_pred_proba) == len(rooms_labels)

    rooms = y_pred_proba > threshold
    rooms_labels_values = [1 if is_room else 0 for is_room in rooms]
    return merge_labels_and_vals_to_dict(rooms_labels, rooms_labels_values)