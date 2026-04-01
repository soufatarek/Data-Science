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
# 3. Data Audit (pre-cleaning quality assessment)
# ---------------------------------------------------------------------------

def data_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a rigorous data quality assessment *before* cleaning.

    This function satisfies the coursework "Data Audit" requirement
    by checking:

    1. **Schema validation** — expected columns, dtypes, value ranges.
    2. **Missing values** — counts and percentages per column.
    3. **Duplicate detection** — exact and user-ID duplicates.
    4. **Outlier detection (IQR)** — for ``sum_gamerounds``.
    5. **Outlier detection (Z-score)** — for all numeric columns.

    Args:
        df: Raw DataFrame from :func:`load_data`.

    Returns:
        Dictionary summarising every check, suitable for rendering
        in a notebook or report.
    """
    from scipy import stats as scipy_stats

    print("=" * 60)
    print("  DATA QUALITY AUDIT")
    print("=" * 60)

    audit: Dict[str, Any] = {}

    # ── 3.1  Schema validation ────────────────────────────────────
    expected_cols = {'userid', 'version', 'sum_gamerounds',
                     'retention_1', 'retention_7'}
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    audit['schema'] = {
        'expected_columns': sorted(expected_cols),
        'actual_columns': sorted(actual_cols),
        'missing_columns': sorted(missing_cols),
        'extra_columns': sorted(extra_cols),
        'schema_valid': len(missing_cols) == 0,
    }
    print(f"\n  Schema valid        : {audit['schema']['schema_valid']}")
    if missing_cols:
        print(f"  ⚠ Missing columns  : {missing_cols}")

    # ── 3.2  Missing values ───────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_pcts = (df.isnull().mean() * 100).round(2)
    audit['missing_values'] = {
        'counts': missing_counts.to_dict(),
        'percentages': missing_pcts.to_dict(),
        'total_missing_cells': int(missing_counts.sum()),
        'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
    }
    print(f"  Total missing cells : {audit['missing_values']['total_missing_cells']}")

    # ── 3.3  Duplicates ───────────────────────────────────────────
    exact_dups = int(df.duplicated().sum())
    userid_dups = int(df['userid'].duplicated().sum()) if 'userid' in df.columns else 0
    audit['duplicates'] = {
        'exact_duplicate_rows': exact_dups,
        'duplicate_userids': userid_dups,
    }
    print(f"  Exact duplicate rows: {exact_dups}")
    print(f"  Duplicate user IDs  : {userid_dups}")

    # ── 3.4  Outlier detection — IQR method ───────────────────────
    if 'sum_gamerounds' in df.columns:
        col = df['sum_gamerounds']
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_iqr = df[(col < lower_bound) | (col > upper_bound)]
        audit['outliers_iqr'] = {
            'column': 'sum_gamerounds',
            'Q1': float(Q1),
            'Q3': float(Q3),
            'IQR': float(IQR),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': len(outliers_iqr),
            'outlier_pct': round(len(outliers_iqr) / len(df) * 100, 2),
        }
        print(f"\n  IQR outliers (sum_gamerounds):")
        print(f"    Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
        print(f"    Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
        print(f"    Outliers: {len(outliers_iqr):,} ({audit['outliers_iqr']['outlier_pct']}%)")

    # ── 3.5  Outlier detection — Z-score ──────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    zscore_outliers: Dict[str, int] = {}
    for c in numeric_cols:
        z = np.abs(scipy_stats.zscore(df[c].dropna()))
        n_outliers = int((z > 3).sum())
        zscore_outliers[c] = n_outliers

    audit['outliers_zscore'] = {
        'threshold': 3.0,
        'outlier_counts': zscore_outliers,
    }
    print(f"\n  Z-score outliers (|z| > 3):")
    for c, n in zscore_outliers.items():
        print(f"    {c:30s}: {n:,}")

    # ── 3.6  Value range checks ───────────────────────────────────
    range_checks: Dict[str, Dict[str, Any]] = {}
    if 'sum_gamerounds' in df.columns:
        range_checks['sum_gamerounds'] = {
            'min': float(df['sum_gamerounds'].min()),
            'max': float(df['sum_gamerounds'].max()),
            'negative_values': int((df['sum_gamerounds'] < 0).sum()),
        }
    if 'version' in df.columns:
        valid_versions = {'gate_30', 'gate_40'}
        actual_versions = set(df['version'].unique())
        range_checks['version'] = {
            'unique_values': sorted(actual_versions),
            'unexpected_values': sorted(actual_versions - valid_versions),
        }
    audit['range_checks'] = range_checks

    print(f"\n  Value ranges:")
    if 'sum_gamerounds' in range_checks:
        r = range_checks['sum_gamerounds']
        print(f"    sum_gamerounds: [{r['min']:.0f}, {r['max']:.0f}]"
              f"  (negative: {r['negative_values']})")

    print(f"\n{'='*60}")
    return audit


# ---------------------------------------------------------------------------
# 4. Cleaning / Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(
    df: pd.DataFrame,
    cap_outliers: bool = True,
    cap_percentile: float = 0.99,
) -> pd.DataFrame:
    """
    Clean the raw dataset.

    Steps performed:
        1. Drop duplicate ``userid`` rows (if any).
        2. Cast boolean retention columns to integers for modelling.
        3. Report any missing values.
        4. **Cap outliers** in ``sum_gamerounds`` at the given percentile
           (default 99th) to reduce extreme-value influence.

    Args:
        df: Raw DataFrame from :func:`load_data`.
        cap_outliers: Whether to cap extreme game-round values.
        cap_percentile: Percentile threshold for capping (default 0.99).

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

    # Outlier capping
    if cap_outliers and 'sum_gamerounds' in df_clean.columns:
        cap_val = df_clean['sum_gamerounds'].quantile(cap_percentile)
        n_capped = (df_clean['sum_gamerounds'] > cap_val).sum()
        df_clean['sum_gamerounds'] = df_clean['sum_gamerounds'].clip(upper=cap_val)
        print(f"Capped {n_capped:,} extreme game-round values at "
              f"{cap_percentile:.0%} percentile ({cap_val:.0f} rounds).")

    print(f"After cleaning: {df_clean.shape[0]:,} rows")
    return df_clean


# ---------------------------------------------------------------------------
# 5. Feature Engineering
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
# 6. A/B Group Splitting
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
# 7. Retention Metrics
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
# 8. Modelling-Ready Split
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
    audit_results = data_audit(df)
    df_clean = preprocess_data(df)
    df_feat = engineer_features(df_clean)
    gate_30, gate_40 = create_ab_groups(df_feat)
    metrics = calculate_retention_metrics(gate_30, gate_40)
    print("\nRetention Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    X_train, X_test, y_train, y_test = prepare_modeling_data(df_feat)