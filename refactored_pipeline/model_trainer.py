"""Model training and evaluation module for Census Income Prediction"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

# Imbalanced learning
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from config import MODEL_PARAMS, RANDOM_STATE, TARGET_COLUMN, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_splits(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/validation sets"""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Data split - Train: {X_train.shape}, Validation: {X_val.shape}")
    logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Target distribution - Val: {y_val.value_counts().to_dict()}")
    
    return X_train, X_val, y_train, y_val


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                            model_params: Dict = None) -> LogisticRegression:
    """Train logistic regression model"""
    logger.info("Training Logistic Regression...")
    
    params = model_params or MODEL_PARAMS['logistic_regression']
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    logger.info("Logistic Regression training completed")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       model_params: Dict = None) -> RandomForestClassifier:
    """Train random forest model"""
    logger.info("Training Random Forest...")
    
    params = model_params or MODEL_PARAMS['random_forest']
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("Random Forest training completed")
    return model


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        # Random Forest style
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    elif hasattr(model, 'coef_'):
        # Logistic Regression style
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
    
    else:
        logger.warning("Model does not have feature importance attributes")
        return pd.DataFrame()
    
    return importance_df


def train_with_sampling_strategy(X_train: pd.DataFrame, y_train: pd.Series, 
                               strategy: str, model_params: Dict = None) -> Any:
    """Train model with specific sampling strategy for class imbalance"""
    
    sampling_strategies = {
        'no_sampling': None,
        'random_oversample': RandomOverSampler(random_state=RANDOM_STATE),
        'smote': SMOTE(random_state=RANDOM_STATE),
        'random_undersample': RandomUnderSampler(random_state=RANDOM_STATE),
        'smoteenn': SMOTEENN(random_state=RANDOM_STATE)
    }
    
    if strategy not in sampling_strategies:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    logger.info(f"Training with sampling strategy: {strategy}")
    
    params = model_params or MODEL_PARAMS['logistic_regression']
    sampler = sampling_strategies[strategy]
    
    if sampler is None:
        # No sampling
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
    else:
        # With sampling
        pipeline = Pipeline([
            ('sampler', sampler),
            ('classifier', LogisticRegression(**params))
        ])
        pipeline.fit(X_train, y_train)
        model = pipeline
    
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Comprehensive model evaluation"""
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }
    
    # ROC AUC and PR AUC if probabilities available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
    })
    
    # Store additional info
    evaluation_results = {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    logger.info(f"Model evaluation completed")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}, ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    
    return evaluation_results


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, 
                        cv_folds: int = 5) -> Dict[str, float]:
    """Perform cross-validation"""
    logger.info("Performing cross-validation...")
    
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")
    
    return cv_results


def save_model(model: Any, filename: str, model_dir: Path = MODELS_DIR) -> Path:
    """Save a trained model"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    filepath = model_dir / filename
    joblib.dump(model, filepath)
    
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath: Path) -> Any:
    """Load a saved model"""
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def compare_models(evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
    """Compare evaluation metrics across multiple models"""
    if not evaluation_results:
        logger.warning("No evaluation results available")
        return pd.DataFrame()
    
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        if 'metrics' in results:
            row = {'model': model_name}
            row.update(results['metrics'])
            comparison_data.append(row)
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by ROC AUC if available, otherwise by accuracy
    sort_col = 'roc_auc' if 'roc_auc' in comparison_df.columns else 'accuracy'
    comparison_df = comparison_df.sort_values(sort_col, ascending=False)
    
    return comparison_df


def train_multiple_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Train multiple models with different configurations"""
    models = {}
    
    # Train baseline models
    logger.info("Training baseline models...")
    models['logistic_regression'] = train_logistic_regression(X_train, y_train)
    models['random_forest'] = train_random_forest(X_train, y_train)
    
    # Train with sampling strategies
    logger.info("Training models with sampling strategies...")
    sampling_strategies = ['no_sampling', 'random_oversample', 'smote', 'random_undersample', 'smoteenn']
    
    for strategy in sampling_strategies:
        model_name = f'lr_{strategy}'
        models[model_name] = train_with_sampling_strategy(X_train, y_train, strategy)
    
    logger.info(f"Trained {len(models)} models total")
    return models


def evaluate_multiple_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
    """Evaluate multiple models and return results"""
    evaluation_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        evaluation_results[model_name] = evaluate_model(model, X_test, y_test)
    
    return evaluation_results


def get_best_model(evaluation_results: Dict[str, Dict], 
                  models: Dict[str, Any], 
                  metric: str = 'roc_auc') -> Tuple[str, Any, Dict]:
    """Get the best performing model based on specified metric"""
    comparison_df = compare_models(evaluation_results)
    
    if comparison_df.empty:
        raise ValueError("No evaluation results available")
    
    # Use accuracy as fallback if specified metric not available
    if metric not in comparison_df.columns:
        logger.warning(f"Metric '{metric}' not available, using 'accuracy' instead")
        metric = 'accuracy'
    
    best_model_name = comparison_df.iloc[0]['model']
    best_model = models[best_model_name]
    best_metrics = evaluation_results[best_model_name]['metrics']
    
    logger.info(f"Best model: {best_model_name} ('{metric}': {best_metrics[metric]:.3f})")
    
    return best_model_name, best_model, best_metrics


def run_complete_training_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Run complete training and evaluation pipeline"""
    logger.info("Starting complete training pipeline...")
    
    # Prepare data splits
    X_train, X_val, y_train, y_val = prepare_data_splits(train_df)
    
    # Prepare test data
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    # Train multiple models
    models = train_multiple_models(X_train, y_train)
    
    # Evaluate on validation set
    val_evaluation_results = evaluate_multiple_models(models, X_val, y_val)
    
    # Get best model
    best_model_name, best_model, best_val_metrics = get_best_model(val_evaluation_results, models)
    
    # Final evaluation on test set
    logger.info(f"Final evaluation of best model ({best_model_name}) on test set...")
    final_test_results = evaluate_model(best_model, X_test, y_test)
    
    # Get feature importance for best model
    feature_names = X_train.columns.tolist()
    feature_importance = get_feature_importance(best_model, feature_names)
    
    # Model comparison
    comparison_df = compare_models(val_evaluation_results)
    
    # Save best model
    model_filename = f"{best_model_name}_best_model.pkl"
    save_model(best_model, model_filename)
    
    results = {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'models': models,
        'validation_results': val_evaluation_results,
        'final_test_metrics': final_test_results['metrics'],
        'comparison': comparison_df,
        'feature_importance': feature_importance,
        'model_saved_to': str(MODELS_DIR / model_filename)
    }
    
    logger.info("Complete training pipeline finished successfully!")
    return results


if __name__ == "__main__":
    # Example usage
    from data_loader import load_and_preprocess_data
    
    try:
        # Load data
        train_df, test_df = load_and_preprocess_data()
        
        # Run pipeline
        results = run_complete_training_pipeline(train_df, test_df)
        
        print("Model Comparison:")
        print(results['comparison'])
        print(f"\nBest Model: {results['best_model_name']}")
        print(f"Final Test Metrics: {results['final_test_metrics']}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")