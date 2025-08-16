import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "housing_price_dataset.csv")

PLOT_STYLE = "default"
COLOR_PALETTE = "husl"
