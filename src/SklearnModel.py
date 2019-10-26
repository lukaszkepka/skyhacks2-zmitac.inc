from ModelBase import ModelBase
import pickle


class SklearnModel(ModelBase):

    def __init__(self, file_name):
        super().__init__(file_name)

    def load_model(self, model_name):
        with open(model_name, 'rb') as f:
            return pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

