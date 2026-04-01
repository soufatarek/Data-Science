"""
Data processing module for Cookie Cats A/B testing project.

This module handles the full data preparation pipeline following
CRISP-DM methodology:
    - Data loading and initial exploration
    - Data cleaning and preprocessing
    - Feature engineering with justifications
    - Train/test splitting for modelling
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the Cookie Cats dataset from CSV.

    The dataset contains ~90,000 player records from an A/B test where
    the first gate was placed at either level 30 (control) or level 40
    (treatment).

    Args:
        data_path: Absolute or relative path to ``cookie_cats.csv``.
            Defaults to ``../data/cookie_cats.csv`` (relative to *src/*).

    Returns:
        Raw DataFrame with columns: userid, version, sum_gamerounds,
        retention_1, retention_7.

    Raises:
        FileNotFoundError: If the CSV cannot be located.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'cookie_cats.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Download it from Kaggle and place it in the data/ directory."
        )

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Exploration helpers
# ---------------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform initial data exploration and return a summary dictionary.

    Useful for the *Data Understanding* phase of CRISP-DM: it gives an
    at-a-glance overview of shape, types, missing values, duplicates,
    and group distributions.

    Args:
        df: Raw DataFrame.

    Returns:
        Dictionary containing shape, column types, missing-value counts,
        summary statistics, unique-value counts, duplicate count, and
        version distribution.
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'summary_stats': df.describe(include='all').to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'duplicates': df.duplicated().sum(),
        'version_distribution': df['version'].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# 3. Cleaning / Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Steps performed:
        1. Drop duplicate ``userid`` rows (if any).
        2. Cast boolean retention columns to integers for modelling.
        3. Report any missing values (the Cookie Cats dataset is
           generally clean, but the check is here for robustness).

    Args:
        df: Raw DataFrame from :func:`load_data`.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df_clean = df.copy()

    # Duplicates
    n_dups = df_clean['userid'].duplicated().sum()
    if n_dups > 0:
        print(f"Dropping {n_dups} duplicate user-ID rows.")
        df_clean = df_clean.drop_duplicates(subset='userid', keep='first')

    # Missing values
    missing = df_clean.isnull().sum()
    if missing.any():
        print("Warning — missing values found:\n", missing[missing > 0])

    # Type casting
    df_clean['retention_1'] = df_clean['retention_1'].astype(int)
    df_clean['retention_7'] = df_clean['retention_7'].astype(int)

    print(f"After cleaning: {df_clean.shape[0]:,} rows")
    return df_clean


# ---------------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to improve model performance.

    Each transformation is justified below:

    ``gamerounds_bin``
        Binning continuous game-rounds into ordinal buckets reduces noise
        and captures non-linear engagement tiers (inactive → casual →
        moderate → hardcore).

    ``high_engagement``
        Binary flag for players whose total rounds exceed the 75th
        percentile.  Heavy players may have fundamentally different
        retention behaviour.

    ``retention_1_x_rounds``
        Interaction term: day-1 retention multiplied by game rounds.
        Captures the combined effect of early return *and* play volume —
        e.g., a player who came back on day 1 **and** played many rounds
        is qualitatively different from one who only did one of the two.

    ``rounds_per_day_proxy``
        A rough engagement-intensity proxy computed as
        ``sum_gamerounds / 7``.  The dataset spans a 7-day window, so
        dividing by 7 approximates daily play frequency.

    Args:
        df: Cleaned DataFrame from :func:`preprocess_data`.

    Returns:
        DataFrame with the original columns plus the engineered features.
    """
    df_feat = df.copy()

    # Binning game rounds into engagement tiers
    bins = [0, 1, 20, 100, 500, float('inf')]
    labels = ['inactive', 'casual', 'moderate', 'active', 'hardcore']
    df_feat['gamerounds_bin'] = pd.cut(
        df_feat['sum_gamerounds'], bins=bins, labels=labels, right=True
    )

    # High-engagement flag (above 75th percentile)
    q75 = df_feat['sum_gamerounds'].quantile(0.75)
    df_feat['high_engagement'] = (df_feat['sum_gamerounds'] > q75).astype(int)

    # Interaction: day-1 retention × total rounds
    df_feat['retention_1_x_rounds'] = (
        df_feat['retention_1'] * df_feat['sum_gamerounds']
    )

    # Engagement-intensity proxy
    df_feat['rounds_per_day_proxy'] = df_feat['sum_gamerounds'] / 7.0

    print(f"Engineered features added. New shape: {df_feat.shape}")
    return df_feat


# ---------------------------------------------------------------------------
# 5. A/B Group Splitting
# ---------------------------------------------------------------------------

def create_ab_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into A/B test groups.

    Args:
        df: Cleaned (and optionally feature-engineered) DataFrame.

    Returns:
        Tuple of (gate_30_df, gate_40_df).
    """
    gate_30 = df[df['version'] == 'gate_30']
    gate_40 = df[df['version'] == 'gate_40']

    print(f"gate_30: {len(gate_30):,}  |  gate_40: {len(gate_40):,}")
    return gate_30, gate_40


# ---------------------------------------------------------------------------
# 6. Retention Metrics
# ---------------------------------------------------------------------------

def calculate_retention_metrics(
    gate_30: pd.DataFrame, gate_40: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate observed retention metrics for both groups.

    Args:
        gate_30: DataFrame for the control group (gate at level 30).
        gate_40: DataFrame for the treatment group (gate at level 40).

    Returns:
        Dictionary with 1-day and 7-day retention rates for each group
        and the observed difference.
    """
    return {
        'retention_30_1day': gate_30['retention_1'].mean(),
        'retention_40_1day': gate_40['retention_1'].mean(),
        'retention_30_7day': gate_30['retention_7'].mean(),
        'retention_40_7day': gate_40['retention_7'].mean(),
        'observed_diff_7day': (
            gate_40['retention_7'].mean() - gate_30['retention_7'].mean()
        ),
    }


# ---------------------------------------------------------------------------
# 7. Modelling-Ready Split
# ---------------------------------------------------------------------------

def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str = 'retention_7',
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix and target vector, then split into train/test.

    The feature set includes:
        * ``sum_gamerounds`` — continuous engagement metric
        * ``version`` — categorical A/B group label
        * ``retention_1`` — binary day-1 return flag
        * ``high_engagement`` — binary flag from feature engineering
        * ``retention_1_x_rounds`` — interaction term
        * ``rounds_per_day_proxy`` — daily engagement proxy

    Args:
        df: Feature-engineered DataFrame.
        target_col: Name of the target column (default ``retention_7``).
        test_size: Fraction reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``
    """
    feature_cols = [
        'sum_gamerounds',
        'version',
        'retention_1',
        'high_engagement',
        'retention_1_x_rounds',
        'rounds_per_day_proxy',
    ]

    # Keep only columns that exist (in case engineer_features was skipped)
    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"Positive-class rate (train): {y_train.mean():.4f}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    info = explore_data(df)
    df_clean = preprocess_data(df)
    df_feat = engineer_features(df_clean)
    gate_30, gate_40 = create_ab_groups(df_feat)
    metrics = calculate_retention_metrics(gate_30, gate_40)
    print("\nRetention Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    X_train, X_test, y_train, y_test = prepare_modeling_data(df_feat)