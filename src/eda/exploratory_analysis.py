import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for prettier plots
def set_plot_style():
    plt.style.use('default')
    sns.set_palette("husl")

# Load the data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def print_overview(df):
    print("Dataset Overview")
    print("=" * 40)
    print(f"Number of houses: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Dataset size: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    print("\nData types:")
    print(df.dtypes)
    print("\nQuick statistics:")
    print(df.describe())

def missing_values_analysis(df):
    print("Checking for missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(1)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        print(f"Found {len(missing_df)} columns with missing values:")
        print(missing_df.to_string(index=False))
        plt.figure(figsize=(10, 4))
        sns.barplot(data=missing_df, x='Percentage', y='Column')
        plt.title('Missing Values by Column')
        plt.xlabel('Percentage Missing (%)')
        plt.tight_layout()
        plt.show()
    else:
        print("No missing values found in the dataset!")

def analyze_target_variable(df):
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    if not price_cols:
        price_cols = [col for col in df.columns if col.lower() in ['cost', 'value', 'amount']]
    if price_cols:
        price_col = price_cols[0]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        price_col = numeric_cols[0]
    print(f"\nAnalyzing target variable: {price_col}")
    print("=" * 40)
    print(f"Average price: {df[price_col].mean():.2f}")
    print(f"Median price: {df[price_col].median():.2f}")
    print(f"Price range: {df[price_col].min():.2f} to {df[price_col].max():.2f}")
    print(f"Standard deviation: {df[price_col].std():.2f}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(df[price_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {price_col}')
    plt.xlabel(price_col)
    plt.ylabel('Count')
    plt.subplot(1, 3, 2)
    plt.boxplot(df[price_col])
    plt.title('Price Box Plot')
    plt.ylabel(price_col)
    plt.subplot(1, 3, 3)
    plt.hist(df[price_col], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Price Distribution (Zoomed)')
    q99 = df[price_col].quantile(0.99)
    plt.xlim(0, q99)
    plt.tight_layout()
    plt.show()
    return price_col

def analyze_categorical_variables(df, price_col):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print("\nCategorical Features Analysis")
        print("=" * 40)
        for col in categorical_cols[:4]:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            print(f"  * {df[col].nunique()} unique values")
            print(f"  * Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            top_values = value_counts.head(8)
            plt.bar(range(len(top_values)), top_values.values)
            plt.xticks(range(len(top_values)), top_values.index, rotation=45)
            plt.title(f'Distribution of {col}')
            plt.ylabel('Count')
            if df[col].nunique() <= 10:
                plt.subplot(1, 2, 2)
                sns.boxplot(x=col, y=price_col, data=df)
                plt.title(f'{price_col} by {col}')
                plt.xticks(rotation=45)
            else:
                plt.subplot(1, 2, 2)
                plt.text(0.5, 0.5, f'Too many categories\n({df[col].nunique()} unique values)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.title('Too many categories for box plot')
            plt.tight_layout()
            plt.show()
    else:
        print("No categorical columns found.")

def analyze_numerical_variables(df, price_col):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if price_col in numeric_cols:
        numeric_cols.remove(price_col)
    print(f"\nNumerical Features Analysis")
    print("=" * 40)
    print(f"Found {len(numeric_cols)} numerical features:")
    print(numeric_cols)
    if numeric_cols:
        print(f"\nStatistics for numerical features:")
        print(df[numeric_cols].describe().round(2))
        n_features = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols[:n_features]):
            axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color='lightcoral')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()
    return numeric_cols

def feature_relationships(df, price_col, numeric_cols):
    print("\nFeature Relationships")
    print("=" * 40)
    numeric_for_corr = [price_col] + numeric_cols
    if len(numeric_for_corr) > 1:
        correlation_matrix = df[numeric_for_corr].corr()
        price_correlations = correlation_matrix[price_col].drop(price_col).abs().sort_values(ascending=False)
        print(f"Features most related to {price_col}:")
        for feature, corr in price_correlations.head(5).items():
            print(f"  * {feature}: {corr:.3f}")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
        top_features = price_correlations.head(4).index.tolist()
        if top_features:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for i, feature in enumerate(top_features):
                axes[i].scatter(df[feature], df[price_col], alpha=0.5)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(price_col)
                axes[i].set_title(f'{price_col} vs {feature}')
                z = np.polyfit(df[feature].dropna(), df[price_col].dropna(), 1)
                p = np.poly1d(z)
                axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
            plt.tight_layout()
            plt.show()

def outlier_detection(df, price_col):
    print("\nOutlier Detection")
    print("=" * 40)
    Q1 = df[price_col].quantile(0.25)
    Q3 = df[price_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[price_col] < lower_bound) | (df[price_col] > upper_bound)]
    print(f"Found {len(outliers)} potential outliers in {price_col}")
    print(f"That's {len(outliers)/len(df)*100:.1f}% of the data")
    if len(outliers) > 0:
        print(f"Outlier price range: {outliers[price_col].min():.2f} to {outliers[price_col].max():.2f}")
        print(f"Normal price range: {lower_bound:.2f} to {upper_bound:.2f}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(df)), df[price_col], alpha=0.6, label='Normal')
    outlier_indices = df[(df[price_col] < lower_bound) | (df[price_col] > upper_bound)].index
    plt.scatter(outlier_indices, df.loc[outlier_indices, price_col], color='red', alpha=0.8, label='Outliers')
    plt.xlabel('House Index')
    plt.ylabel(price_col)
    plt.title('Houses with Outlier Prices')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.boxplot(df[price_col])
    plt.title(f'{price_col} Box Plot')
    plt.ylabel(price_col)
    plt.tight_layout()
    plt.show()
