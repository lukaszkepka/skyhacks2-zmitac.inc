from ModelBase import ModelBase
from keras.models import load_model


class KerasModel(ModelBase):

    def __init__(self, file_name):
        super().__init__(file_name)

    def load_model(self, model_name):
        return load_model(model_name)
