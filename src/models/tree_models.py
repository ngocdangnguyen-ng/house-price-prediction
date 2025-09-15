
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .base_model import BaseModel

class TreeModel(BaseModel):
    def __init__(self, model_type="random_forest", **kwargs):
        if model_type == "decision_tree":
            self.model = DecisionTreeRegressor(**kwargs)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(**kwargs)
        else:
            self.model = RandomForestRegressor(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
