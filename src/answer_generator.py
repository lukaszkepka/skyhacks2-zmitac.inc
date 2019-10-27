import csv
import logging
import os
import random
from typing import Tuple

import numpy as np

import utils
from KerasModel import KerasModel
from ModelBase import ModelBase
from NaiveModel import NaiveModel
from SklearnModel import SklearnModel

__author__ = 'ING_DS_TECH'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)


input_dir = "../Images"
models_dir = "../Models"
answers_file = "../answers.csv"
annotations_file = "../labels.csv"

img_shape = (224, 224)

labels_all_in_order = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                       'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                       'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                       'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                       'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                       'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                       'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                       'Wall', 'Window']

rooms_labels = ['Bathroom', 'Bedroom', 'Dining room', 'House', 'Kitchen', 'Living room']
rooms_labels_task_2 = ['bathroom', 'bedroom', 'dinning_room', 'house', 'kitchen', 'living_room']

features_selected_knn = ['Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Cabinetry',
                         'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers', 'Coffee table',
                         'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Door', 'Drawer',
                         'Facade', 'Floor', 'Furniture', 'Grass', 'Hardwood',
                         'Kitchen & dining room table', 'Kitchen stove', 'Mattress',
                         'Nightstand', 'Plumbing fixture', 'Property', 'Real estate',
                         'Refrigerator', 'Roof', 'Room', 'Rural area', 'Shower', 'Sink', 'Sky',
                         'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                         'Wall']

features_selected_rf = ['Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                        'Bed sheet', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier',
                        'Chest of drawers', 'Coffee table', 'Couch', 'Countertop', 'Cupboard',
                        'Curtain', 'Door', 'Drawer', 'Facade', 'Fireplace', 'Floor',
                        'Furniture', 'Grass', 'Hardwood', 'Kitchen stove', 'Mattress',
                        'Nightstand', 'Plumbing fixture', 'Property', 'Real estate',
                        'Refrigerator', 'Roof', 'Room', 'Shower', 'Sink', 'Sky', 'Table', 'Tap',
                        'Tile', 'Toilet', 'Tree', 'Urban area', 'Wall', 'Window']

filtered_labels = [label for label in labels_all_in_order if label not in rooms_labels]

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]

output = []

selection = True
models = {'task_1': KerasModel(os.path.join(models_dir, 'Xception.h5')),
          'task_2': SklearnModel(os.path.join(models_dir, 'trained_rf_selection.sav')),
          'task_3_tech': KerasModel(os.path.join(models_dir, 'MobileNet_tech_new.h5')),
          'task_3_std': KerasModel(os.path.join(models_dir, 'MobileNet_std_new.h5'))}


def task_1(file_path: str, model: ModelBase) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))

    labels_values_pred = np.round(model.predict(utils.load_input_img(file_path, img_shape))[0])

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return utils.merge_labels_and_vals_to_dict(filtered_labels, labels_values_pred)


def task_2(output_per_file: dict, file_path: str, model: ModelBase):
    logger.debug("Performing task 2 for file {0}".format(file_path))

    labels_used = filtered_labels
    if selection:
        labels_used = [label for label in labels_used if label in features_selected_rf]
    X = [output_per_file[label] for label in labels_used]
    X = np.expand_dims(X, axis=0)
    y_pred_proba = model.predict_proba(X)[0]
    output_per_file.update(utils.predict_rooms_labels(y_pred_proba, rooms_labels))
    y_pred = model.predict(X)[0]
    output_per_file = dict((k, int(v)) for k, v in output_per_file.items())
    output_per_file['task2_class'] = rooms_labels_task_2[y_pred]

    logger.debug("Done with Task 2 for file {0}".format(file_path))
    return output_per_file


def task_3(file_path: str, model: ModelBase):
    logger.debug("Performing task 3 for file {0}".format(file_path))
    y_pred = model.predict(utils.load_input_img(file_path, img_shape))[0]
    print(y_pred)
    result = np.argmax(y_pred) + 1
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return result


def main():
    logger.debug("Sample answers file generator")
    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                output_per_file = task_1(file_path, models['task_1'])
                output_per_file = task_2(output_per_file, file_path, models['task_2'])
                output_per_file['tech_cond'] = task_3(file_path, models['task_3_tech'])
                output_per_file['standard'] = task_3(file_path, models['task_3_std'])
                output_per_file['filename'] = f
                output.append(output_per_file)

    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_all_in_order)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
