import pandas as pd
import numpy as np
from typing import List, Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Load the housing dataset from a CSV file and strip column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def create_domain_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create domain-specific features (ratios, densities, age, location, price per unit).
    Returns the modified DataFrame and list of new feature names.
    """
    new_features = []
    # Living space ratios
    living_cols = [col for col in df.columns if 'sqft' in col.lower() or 'area' in col.lower()]
    if len(living_cols) >= 2:
        for i, col1 in enumerate(living_cols):
            for col2 in living_cols[i+1:]:
                if col1 != col2 and not col1.endswith('_scaled') and not col2.endswith('_scaled'):
                    ratio_name = f"{col1}_to_{col2}_ratio"
                    df[ratio_name] = df[col1] / (df[col2] + 1)
                    new_features.append(ratio_name)
    # Room density
    room_cols = [col for col in df.columns if 'room' in col.lower() or 'bedroom' in col.lower() or 'bathroom' in col.lower()]
    area_cols = [col for col in df.columns if 'sqft' in col.lower() and 'living' in col.lower()]
    if room_cols and area_cols:
        for room_col in room_cols:
            for area_col in area_cols:
                if not room_col.endswith('_scaled') and not area_col.endswith('_scaled'):
                    density_name = f"{room_col}_per_sqft"
                    df[density_name] = df[room_col] / (df[area_col] + 1)
                    new_features.append(density_name)
    # Age features
    year_cols = [col for col in df.columns if 'year' in col.lower() or 'built' in col.lower()]
    for col in year_cols:
        if not col.endswith('_scaled') and np.issubdtype(df[col].dtype, np.number):
            age_col = f"{col}_age"
            df[age_col] = 2024 - df[col]
            new_features.append(age_col)
            age_cat_col = f"{col}_category"
            df[age_cat_col] = pd.cut(df[age_col], bins=[0, 10, 25, 50, 100, float('inf')], labels=['New', 'Recent', 'Mature', 'Old', 'Historic'])
            new_features.append(age_cat_col)
    # Location features
    location_cols = [col for col in df.columns if any(word in col.lower() for word in ['zip', 'location', 'city', 'neighborhood'])]
    for col in location_cols:
        if df[col].dtype == 'object' or col.endswith('_encoded'):
            location_avg_col = f"{col}_avg_price"
            location_avg = df.groupby(col)[target_col].mean()
            df[location_avg_col] = df[col].map(location_avg)
            new_features.append(location_avg_col)
    # Price per unit
    size_cols = [col for col in df.columns if any(word in col.lower() for word in ['sqft', 'area', 'size'])]
    for col in size_cols:
        if not col.endswith('_scaled') and np.issubdtype(df[col].dtype, np.number):
            price_per_unit = f"price_per_{col}"
            df[price_per_unit] = df[target_col] / (df[col] + 1)
            new_features.append(price_per_unit)
    return df, new_features

def create_interaction_features(df: pd.DataFrame, top_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create pairwise interaction features (multiplication, addition, ratio) for top features.
    Returns the modified DataFrame and list of new feature names.
    """
    interaction_features = []
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if feat1 != feat2:
                mult_name = f"{feat1}_x_{feat2}"
                df[mult_name] = df[feat1] * df[feat2]
                interaction_features.append(mult_name)
                add_name = f"{feat1}_plus_{feat2}"
                df[add_name] = df[feat1] + df[feat2]
                interaction_features.append(add_name)
                if (df[feat2] > 0).all():
                    ratio_name = f"{feat1}_div_{feat2}"
                    df[ratio_name] = df[feat1] / df[feat2]
                    interaction_features.append(ratio_name)
    return df, interaction_features

def create_math_transformations(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create polynomial, sqrt, and log transformations for selected features.
    Returns the modified DataFrame and list of new feature names.
    """
    transformation_features = []
    for col in features:
        if (df[col] >= 0).all():
            square_name = f"{col}_squared"
            df[square_name] = df[col] ** 2
            transformation_features.append(square_name)
            sqrt_name = f"{col}_sqrt"
            df[sqrt_name] = np.sqrt(df[col])
            transformation_features.append(sqrt_name)
        if (df[col] > 0).all():
            log_name = f"{col}_log"
            df[log_name] = np.log(df[col])
            transformation_features.append(log_name)
    return df, transformation_features

def create_binned_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create quantile-based binned categorical features for selected continuous features.
    Returns the modified DataFrame and list of new feature names.
    """
    binning_features = []
    for col in features:
        if np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 10:
            binned_name = f"{col}_binned"
            df[binned_name] = pd.qcut(df[col], q=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
            binning_features.append(binned_name)
    return df, binning_features

def select_features(df: pd.DataFrame, target_col: str, k: int = 20) -> List[str]:
    """
    Select top k features using correlation and SelectKBest.
    Returns the list of selected feature names.
    """
    from sklearn.feature_selection import SelectKBest, f_regression
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)
    # Correlation
    feature_correlations = df[numeric_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    top_corr = feature_correlations.head(k).index.tolist()
    # SelectKBest
    selector = SelectKBest(score_func=f_regression, k=min(k, len(numeric_features)))
    X_selected = selector.fit_transform(df[numeric_features], df[target_col])
    selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
    # Combine and deduplicate
    final_features = list(set(top_corr + selected_features))
    return final_features
import pandas as pd
