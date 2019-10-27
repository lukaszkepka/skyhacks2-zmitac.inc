import os
import shutil
import pandas as pd
from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
from sklearn.utils import shuffle

dataset_path = "D:\\Datasets\\skyhacks\\main_task_data\\"
annotations_path = 'D:\\Programowanie\\zmitac.inc\\main\\labels.csv'

label_dict = {'house' : 'House',
'dining_room' : 'Dining room',
'kitchen' : 'Kitchen',
'bathroom' : 'Bathroom',
'living_room' : 'Living room',
'bedroom' : 'Bedroom'}


# Load annotations dataframe
def read_annotations(annotations_path):
    labels_to_remove = ['Bathroom', 'Bedroom', 'Dining room', 'House', 'Kitchen', 'Living room', 'standard', 'filename']
    df = pd.read_csv(annotations_path)

    rows_to_delete = []
    for index, row in df.iterrows():
        if row['task2_class'] == 'validation':
            rows_to_delete.append(index)

    df = df.drop(rows_to_delete)

    for index, row in df.iterrows():
        row['Bathroom'] = 0
        row['Bedroom'] = 0
        row['Dining room'] = 0
        row['House'] = 0
        row['Kitchen'] = 0
        row['Living room'] = 0
        row[label_dict[row['task2_class']]] = 1

    df = df.drop(['standard', 'filename', 'task2_class'], axis=1)
    return df

def to_one_hot(array):
    res = np.zeros(shape=(len(array), 4))
    for i, row in enumerate(array):
        res[i, row - 1] = 1

    return res


df = read_annotations(annotations_path)
df = shuffle(df)
train_y = df.iloc[:, df.columns.to_list().index('tech_cond')].to_numpy()
train_y = to_one_hot(train_y)
df = df.drop(['tech_cond'], axis=1)
train_x = df.to_numpy()

val_x = train_x[0:200]
val_y = train_y[0:200]
train_x = train_x[200:]
train_y = train_y[200:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_x.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x,
                    train_y,
                    epochs=100,
                    batch_size=4,
                    validation_data=(val_x, val_y))


wajcha = True

#
# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)
#
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)
#
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
#                              train_data[0]])
#
# model = models.Sequential()
# model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(46, activation='softmax'))
#
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# validation_set = x_train[:1000]
# partial_x_train = x_train[1000:]
#
# validation_labels = one_hot_train_labels[:1000]
# partial_x_labels = one_hot_train_labels[1000:]
#
# history = model.fit(partial_x_train,
#                     partial_x_labels,
#                     epochs=9,
#                     batch_size=512,
#                     validation_data=(validation_set, validation_labels))
