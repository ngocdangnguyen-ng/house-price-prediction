import joblib
from sklearn.neural_network import MLPRegressor
from .base_model import BaseModel

class NeuralNetModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
