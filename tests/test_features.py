import pandas as pd
import numpy as np
import os
import pytest
from src.features import feature_engineering

def test_create_domain_features():
	data_path = os.path.join('data', 'processed', 'cleaned_data.csv')
	df = pd.read_csv(data_path)
	target_col = [c for c in df.columns if 'price' in c.lower()][0]
	df2, new_feats = feature_engineering.create_domain_features(df, target_col)
	assert isinstance(df2, pd.DataFrame)
	assert isinstance(new_feats, list)
