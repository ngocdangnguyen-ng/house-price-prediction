
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from .base_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self, model_type="linear", **kwargs):
        if model_type == "ridge":
            self.model = Ridge(**kwargs)
        elif model_type == "lasso":
            self.model = Lasso(**kwargs)
        else:
            self.model = LinearRegression(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)