import csv
import os
import logging
import random
from typing import Tuple

from KerasModel import KerasModel
from ModelBase import ModelBase
from SklearnModel import SklearnModel

__author__ = 'ING_DS_TECH'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = "main_task_data"
answers_file = "answers.csv"
models_dir = "Models"

labels_all_in_order = ['Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed', 'Bed frame',
                       'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
                       'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
                       'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
                       'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
                       'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
                       'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
                       'Wall', 'Window']

labels_to_skip = ['Bathroom', 'Bedroom', 'Dining room', 'House', 'Kitchen', 'Living room']

labels_task2 = ['apartment', 'bathroom', 'bedroom', 'dinning_room', 'house', 'kitchen', 'living_room']

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]

output = []

models = {'task_1': KerasModel(os.path.join(models_dir, 'DenseNet121_2.h5')),
          'task_2': SklearnModel(os.path.join(models_dir, 'DenseNet121_2.h5')),
          'task_3': KerasModel(os.path.join(models_dir, 'DenseNet121_2.h5')),
          }


def task_1(partial_output: dict, file_path: str, model: ModelBase) -> dict:
    logger.debug("Performing task 1 for file {0}".format(file_path))

    for label in labels_all_in_order:
        if label not in labels_to_skip:
            partial_output[label] = 0

    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return partial_output


def task_2(file_path: str, model: ModelBase) -> str:
    logger.debug("Performing task 2 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return labels_task2[random.randrange(len(labels_task2))]


def task_3(file_path: str, model: ModelBase) -> Tuple[str, str]:
    logger.debug("Performing task 3 for file {0}".format(file_path))
    #
    #
    #	HERE SHOULD BE A REAL SOLUTION
    #
    #
    logger.debug("Done with Task 1 for file {0}".format(file_path))
    return labels_task3_1[random.randrange(len(labels_task3_1))], labels_task3_2[random.randrange(len(labels_task3_2))]


def main():
    logger.debug("Sample answers file generator")
    for dirpath, dnames, fnames in os.walk(input_dir):
        for f in fnames:
            if f.endswith(".jpg"):
                file_path = os.path.join(dirpath, f)
                output_per_file = {'filename': f,
                                   'task2_class': task_2(file_path, models['task_2']),
                                   'tech_cond': task_3(file_path, models['task_3'])[0],
                                   'standard': task_3(file_path, models['task_3'])[1]
                                   }
                output_per_file = task_1(output_per_file, file_path, models['task_1'])

                output.append(output_per_file)

    with open(answers_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['filename', 'standard', 'task2_class', 'tech_cond'] + labels_task_1)
        writer.writeheader()
        for entry in output:
            logger.debug(entry)
            writer.writerow(entry)


if __name__ == "__main__":
    main()
