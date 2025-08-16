import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Dict, Any, Tuple

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values: fill categorical with mode/'Unknown', numeric with median.
    """
    missing = df.isnull().sum()
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing,
        'Missing_Percentage': (missing / len(df)) * 100
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    for _, row in missing_summary.iterrows():
        col = row['Column']
        missing_pct = row['Missing_Percentage']
        if df[col].dtype == 'object':
            if missing_pct < 50:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
            else:
                df[col] = df[col].fillna('Unknown')
        else:
            if missing_pct < 50:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            # else: consider dropping column (not done here)
    return df

def detect_and_treat_outliers(df: pd.DataFrame, price_col: str = None) -> pd.DataFrame:
    """
    Detect and optionally remove outliers in the price column using IQR.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not price_col:
        for col in ['price', 'Price', 'price_per_m2', 'sale_price']:
            if col in df.columns:
                price_col = col
                break
        if not price_col and numeric_cols:
            price_col = numeric_cols[0]
    if price_col:
        Q1 = df[price_col].quantile(0.25)
        Q3 = df[price_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask = (df[price_col] < lower_bound) | (df[price_col] > upper_bound)
        outliers_count = outliers_mask.sum()
        if outliers_count <= len(df) * 0.1:
            df = df[~outliers_mask].copy()
    return df

def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric-looking strings to numbers, and parse dates.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].str.contains(r'^[\d\.\,\-\+]+$', regex=True, na=False).any():
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='ignore')
                except Exception:
                    pass
    date_keywords = ['date', 'year', 'time']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
    return df

def create_basic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create simple derived features based on common house price factors.
    """
    features_created = []
    if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        features_created.append('total_rooms')
    if 'sqft_living' in df.columns and 'price' in df.columns:
        df['price_per_sqft'] = df['price'] / df['sqft_living']
        features_created.append('price_per_sqft')
    if 'yr_built' in df.columns:
        df['house_age'] = 2024 - df['yr_built']
        features_created.append('house_age')
    if 'sqft_lot' in df.columns and 'sqft_living' in df.columns:
        df['lot_to_living_ratio'] = df['sqft_lot'] / df['sqft_living']
        features_created.append('lot_to_living_ratio')
    return df, features_created

def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode categorical columns using label encoding or one-hot encoding.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count == 2:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
        elif unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
        else:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    return df, categorical_cols

def scale_numeric_features(df: pd.DataFrame, exclude: List[str] = None) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Scale numeric features (excluding encoded categorical columns).
    """
    if exclude is None:
        exclude = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_for_scaling = [col for col in numeric_cols if col not in exclude]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols_for_scaling])
    scaled_df = pd.DataFrame(scaled_data, columns=[f"{col}_scaled" for col in numeric_cols_for_scaling], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    return df, numeric_cols_for_scaling, scaler