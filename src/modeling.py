"""
Machine Learning module for Cookie Cats retention prediction.

Implements an ``sklearn.pipeline.Pipeline``-based workflow for binary
classification (retained vs. churned at day 7).  The module provides:

    * Preprocessing (scaling + encoding) via ``ColumnTransformer``
    * Class-imbalance handling via SMOTE
    * Multiple model training  (Logistic Regression, Random Forest,
      XGBoost, Gradient Boosting)
    * Comprehensive evaluation metrics (accuracy, precision, recall,
      F1, ROC-AUC) with written justifications
    * Hyperparameter tuning via ``GridSearchCV``
    * Model persistence with ``joblib``
"""

import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ── Metric explanation texts (for notebook rendering) ─────────────────────

METRIC_EXPLANATIONS: Dict[str, str] = {
    "accuracy": (
        "**Accuracy** measures the fraction of correct predictions overall. "
        "It is intuitive but can be misleading when classes are imbalanced "
        "(e.g., if 80 % of players churn, always predicting churn yields 80 % accuracy)."
    ),
    "precision": (
        "**Precision** answers *'Of all players the model predicted would be "
        "retained, how many actually were?'*  High precision minimises "
        "false positives — important when the cost of a wrong positive "
        "prediction is high."
    ),
    "recall": (
        "**Recall (Sensitivity)** answers *'Of all players who were actually "
        "retained, how many did the model correctly identify?'*  High recall "
        "minimises false negatives — critical when missing a retained player "
        "has consequences (e.g., withholding an incentive they deserved)."
    ),
    "f1": (
        "**F1-score** is the harmonic mean of precision and recall.  It "
        "provides a single balanced metric, especially useful when the class "
        "distribution is uneven and neither precision nor recall alone "
        "tells the full story."
    ),
    "roc_auc": (
        "**ROC-AUC** measures the model's ability to discriminate between "
        "classes across all probability thresholds.  AUC = 0.5 is random "
        "guessing; AUC = 1.0 is perfect separation.  It is threshold-"
        "independent and well-suited for comparing classifiers."
    ),
}


# ── Preprocessing ─────────────────────────────────────────────────────────

def create_preprocessor(
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` that scales numeric features and
    one-hot-encodes categorical features.

    Args:
        numeric_features: List of numeric column names.
            Defaults to ``['sum_gamerounds', 'retention_1',
            'high_engagement', 'retention_1_x_rounds',
            'rounds_per_day_proxy']``.
        categorical_features: List of categorical column names.
            Defaults to ``['version']``.

    Returns:
        Fitted-ready ``ColumnTransformer``.
    """
    if numeric_features is None:
        numeric_features = [
            'sum_gamerounds',
            'retention_1',
            'high_engagement',
            'retention_1_x_rounds',
            'rounds_per_day_proxy',
        ]
    if categorical_features is None:
        categorical_features = ['version']

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ],
        remainder='drop',
    )


# ── Pipeline construction ─────────────────────────────────────────────────

def build_pipeline(
    model,
    preprocessor: ColumnTransformer = None,
    use_smote: bool = True,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Wrap preprocessing, optional SMOTE, and a classifier into a single
    ``imblearn.pipeline.Pipeline`` (which is fully compatible with
    ``sklearn.pipeline.Pipeline`` but additionally supports resamplers).

    Args:
        model: An sklearn-compatible estimator instance.
        preprocessor: A ``ColumnTransformer``.  If *None* a default one
            is created via :func:`create_preprocessor`.
        use_smote: Whether to add a SMOTE step for class balancing.
        random_state: Seed for SMOTE.

    Returns:
        An ``ImbPipeline`` ready for ``.fit()`` / ``.predict()``.
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()

    steps = [('preprocessor', preprocessor)]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('classifier', model))

    return ImbPipeline(steps)


# ── Model training ────────────────────────────────────────────────────────

def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    Return a dictionary of candidate classifiers.

    Returns:
        ``{name: estimator_instance}``
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=random_state, max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=random_state
        ),
        'XGBoost': XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            device='cuda',
            tree_method='hist',
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=random_state
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train every model in the zoo inside its own pipeline.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: Optional custom preprocessor.
        random_state: Seed.

    Returns:
        ``{model_name: {'pipeline': fitted_pipeline}}``
    """
    zoo = get_model_zoo(random_state)
    results: Dict[str, Dict[str, Any]] = {}

    for name, estimator in zoo.items():
        print(f"  Training {name} …")
        pipe = build_pipeline(estimator, preprocessor, random_state=random_state)
        pipe.fit(X_train, y_train)
        results[name] = {'pipeline': pipe}

    return results


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate a single pipeline on the test set and return all metrics.

    Metrics returned: accuracy, precision, recall, F1, ROC-AUC,
    classification report (text), and confusion matrix.

    Args:
        pipeline: A fitted pipeline.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dictionary of metric values.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_proba,
    }


def evaluate_all_models(
    trained: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate every trained model and print results.

    Args:
        trained: Output of :func:`train_models`.
        X_test: Test features.
        y_test: True labels.

    Returns:
        ``{model_name: {metric: value, …}}``
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, info in trained.items():
        metrics = evaluate_model(info['pipeline'], X_test, y_test)
        results[name] = metrics
        print(f"\n{'='*50}")
        print(f" {name}")
        print(f"{'='*50}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1-score  : {metrics['f1']:.4f}")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    return results


def get_best_model(
    evaluation_results: Dict[str, Dict[str, Any]],
    metric: str = 'roc_auc',
) -> Tuple[str, float]:
    """
    Identify the best model based on the chosen metric.

    Args:
        evaluation_results: Output of :func:`evaluate_all_models`.
        metric: Metric name to rank by (default ``roc_auc``).

    Returns:
        ``(best_model_name, best_score)``
    """
    best_name = max(evaluation_results, key=lambda n: evaluation_results[n][metric])
    best_score = evaluation_results[best_name][metric]
    print(f"\n★ Best model by {metric}: {best_name} ({best_score:.4f})")
    return best_name, best_score


def metrics_summary_df(
    evaluation_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Return a tidy DataFrame comparing all models across key metrics.

    Useful for rendering a comparison table in a notebook.
    """
    rows = []
    for name, m in evaluation_results.items():
        rows.append({
            'Model': name,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1': m['f1'],
            'ROC-AUC': m['roc_auc'],
        })
    return pd.DataFrame(rows).set_index('Model').sort_values('ROC-AUC', ascending=False)


# ── Hyperparameter Tuning ─────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = 'XGBoost',
    preprocessor: ColumnTransformer = None,
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run ``GridSearchCV`` with cross-validation to find optimal
    hyperparameters.

    Tuning grids (≥ 3 parameters each):

    * **XGBoost**: ``n_estimators``, ``max_depth``, ``learning_rate``
    * **Random Forest**: ``n_estimators``, ``max_depth``,
      ``min_samples_split``

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_name: ``'XGBoost'`` or ``'Random Forest'``.
        preprocessor: Optional custom ``ColumnTransformer``.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for ``GridSearchCV``.
        random_state: Seed.

    Returns:
        Dictionary with ``best_params``, ``best_cv_score``, and
        ``best_pipeline``.

    Raises:
        ValueError: If ``model_name`` is unsupported.
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()

    if model_name == 'XGBoost':
        base_pipe = build_pipeline(
            XGBClassifier(random_state=random_state, eval_metric='logloss',
                          use_label_encoder=False, device='cuda',
                          tree_method='hist'),
            preprocessor, random_state=random_state,
        )
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
        }
    elif model_name == 'Random Forest':
        base_pipe = build_pipeline(
            RandomForestClassifier(random_state=random_state),
            preprocessor, random_state=random_state,
        )
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
        }
    else:
        raise ValueError(f"Tuning not implemented for '{model_name}'")

    print(f"Tuning {model_name} ({cv}-fold CV, scoring={scoring}) …")
    gs = GridSearchCV(
        base_pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV {scoring}: {gs.best_score_:.4f}")

    return {
        'best_params': gs.best_params_,
        'best_cv_score': gs.best_score_,
        'best_pipeline': gs.best_estimator_,
    }


# ── Visualisation helpers ─────────────────────────────────────────────────

def plot_model_comparison(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: str = None,
) -> None:
    """Bar chart comparing ROC-AUC scores across models."""
    names = list(evaluation_results.keys())
    scores = [evaluation_results[n]['roc_auc'] for n in names]

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=names, y=scores, palette='viridis')
    plt.title('Model Comparison — ROC-AUC', fontsize=14)
    plt.ylabel('ROC-AUC Score')
    plt.ylim(0.45, 1.0)
    for i, s in enumerate(scores):
        plt.text(i, s + 0.01, f'{s:.4f}', ha='center', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(
    evaluation_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: str = None,
) -> None:
    """Overlay ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, m in evaluation_results.items():
        fpr, tpr, _ = roc_curve(y_test, m['probabilities'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrices(
    evaluation_results: Dict[str, Dict[str, Any]],
    save_path: str = None,
) -> None:
    """Plot confusion matrices for every model side-by-side."""
    n = len(evaluation_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, evaluation_results.items()):
        sns.heatmap(m['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=ax, cbar=False)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Model persistence ─────────────────────────────────────────────────────

def save_model(pipeline, path: str) -> None:
    """Persist a fitted pipeline to disk with ``joblib``."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")


def load_model(path: str):
    """Load a previously saved pipeline."""
    return joblib.load(path)


# ── CLI entry-point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from processing import load_data, preprocess_data, engineer_features, prepare_modeling_data

    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = prepare_modeling_data(df)

    trained = train_models(X_train, y_train)
    results = evaluate_all_models(trained, X_test, y_test)
    print(metrics_summary_df(results))
    best_name, _ = get_best_model(results)

    tuned = tune_hyperparameters(X_train, y_train, 'XGBoost')
    tuned_metrics = evaluate_model(tuned['best_pipeline'], X_test, y_test)
    print(f"\nTuned XGBoost ROC-AUC: {tuned_metrics['roc_auc']:.4f}")