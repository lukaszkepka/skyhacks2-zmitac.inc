import os
import cv2
import pandas as pd


def list_all_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    return files

dataset_path = "D:\\Datasets\\skyhacks\\main_task_data"
annotations_path = "D:\\Programowanie\\zmitac.inc\\Models\\labels.csv"
annotations = pd.read_csv(annotations_path)
annotations = annotations.drop(['tech_cond', 'standard', 'task2_class'], axis=1)
for file_path in list_all_files(dataset_path):
    contained_objects = []
    image = cv2.imread(file_path)
    file_name = os.path.basename(file_path)
    annotation = annotations.loc[annotations['filename'] == file_name].head(1)
    if not annotation.empty:
        for i, column in enumerate(annotations.columns.to_list()):
            if str(annotation.iloc[0, i]) == '1':
                contained_objects.append(column)
    print(contained_objects)
    image = cv2.resize(image, (640, 480))
    cv2.imshow('Image', image)
    cv2.waitKey(0)