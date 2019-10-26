from builtins import range

from ModelBase import ModelBase
from random import choices
import pandas as pd
import pickle


class NaiveModel(ModelBase):

    def __init__(self, model_name):
        super().__init__('')
        self.load_annotations(model_name)

    def load_model(self, model_name):
        return None

    def load_annotations(self, annotations_path):
        self.annotations = pd.read_csv(annotations_path)

    def compute_distribution_for_column(self, column_name):
        column_index = self.annotations.columns.to_list().index(column_name)
        labels = self.annotations.iloc[:, column_index]

        hist = {}
        for i in range(labels.min(), labels.max()):
            hist[i] = 0

        for label in labels:
            if label not in hist:
                hist[label] = 0
            hist[label] += 1

        for key in hist.keys():
            hist[key] = hist[key] / len(labels)

        return hist

    def predict_for_column(self, column):
        an = v.compute_distribution_for_column(column)
        population = list(an.keys())
        weights = list(an.values())
        return choices(population, weights)

    def predict(self, X=None):
        return (self.predict_for_column('tech_cond'), self.predict_for_column('standard'))

    def predict_proba(self, X):
        return None
