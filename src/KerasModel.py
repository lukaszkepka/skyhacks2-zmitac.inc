from keras.models import load_model as keras_load_model
import numpy as np

from src.ModelBase import ModelBase


class KerasModel(ModelBase):

    def __init__(self, file_name):
        super().__init__(file_name)

    def load_model(self, model_name):
        return keras_load_model(model_name)

    def predict(self, X):
        """ X is image """
        return self.model.predict(X)
