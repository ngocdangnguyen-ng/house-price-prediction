import pandas as pd
import os
import pytest
from src.models.linear_models import LinearModel

def test_linear_model_train_predict():
	data_path = os.path.join('data', 'processed', 'cleaned_data.csv')
	df = pd.read_csv(data_path)
	target_col = [c for c in df.columns if 'price' in c.lower()][0]
	features = [c for c in df.columns if c != target_col]
	X = df[features].select_dtypes(include=[float, int])
	y = df[target_col]
	model = LinearModel()
	model.train(X, y)
	preds = model.predict(X)
	assert len(preds) == len(y)

def test_linear_model_save_load(tmp_path):
	data_path = os.path.join('data', 'processed', 'cleaned_data.csv')
	df = pd.read_csv(data_path)
	target_col = [c for c in df.columns if 'price' in c.lower()][0]
	features = [c for c in df.columns if c != target_col]
	X = df[features].select_dtypes(include=[float, int])
	y = df[target_col]
	model = LinearModel()
	model.train(X, y)
	save_path = tmp_path / 'test_model.pkl'
	model.save(str(save_path))
	model2 = LinearModel()
	model2.load(str(save_path))
	preds2 = model2.predict(X)
	assert len(preds2) == len(y)
