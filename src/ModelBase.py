import abc


class ModelBase(metaclass=abc.ABCMeta):

    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    @abc.abstractmethod
    def load_model(self, model_name):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass