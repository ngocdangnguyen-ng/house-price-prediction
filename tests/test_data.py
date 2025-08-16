import pandas as pd
import os
import pytest
from src.data.data_loader import load_data

def test_load_data():
	data_path = os.path.join('data', 'raw', 'housing_price_dataset.csv')
	df = load_data(data_path)
	assert isinstance(df, pd.DataFrame)
	assert df.shape[0] > 0
	assert any('price' in c.lower() for c in df.columns)
