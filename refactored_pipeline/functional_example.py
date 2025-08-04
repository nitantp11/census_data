"""Example of functional approach to the same pipeline"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict, Any

from config import COLUMN_NAMES, TARGET_COLUMN, FEATURES_TO_DROP

# ========================================
# FUNCTIONAL APPROACH - SIMPLER & CLEANER
# ========================================

def load_census_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and assign column names to census data"""
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    train_df.columns = COLUMN_NAMES
    test_df.columns = COLUMN_NAMES
    
    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw census data"""
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.replace('?', np.nan)
    
    # Clean target variable
    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].str.strip()
        df_clean[TARGET_COLUMN] = (df_clean[TARGET_COLUMN] == '50000+').astype(int)
    
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features"""
    df_eng = df.copy()
    
    # Age groups
    if 'age' in df_eng.columns:
        df_eng['age_group'] = pd.cut(
            df_eng['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
    
    # Financial indicators
    if 'capital_gains' in df_eng.columns:
        df_eng['has_capital_gain'] = (df_eng['capital_gains'] > 0).astype(int)
    
    return df_eng


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Final feature preparation for modeling"""
    df_prep = df.copy()
    
    # Drop irrelevant columns
    drop_cols = [col for col in FEATURES_TO_DROP if col in df_prep.columns]
    df_prep = df_prep.drop(columns=drop_cols)
    
    # Handle missing values
    df_prep = df_prep.fillna(df_prep.median())
    
    # One-hot encode categorical variables
    categorical_cols = df_prep.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
    
    if len(categorical_cols) > 0:
        df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)
    
    return df_prep


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = 'logistic') -> Any:
    """Train a single model"""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    return model


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate model performance"""
    y_pred = model.predict(X)
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'classification_report': classification_report(y, y_pred)
    }


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target"""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


# ========================================
# SIMPLE PIPELINE COMPOSITION
# ========================================

def run_simple_pipeline(train_path: str, test_path: str) -> Dict[str, Any]:
    """Run complete pipeline using functional composition"""
    
    # Load data
    train_df, test_df = load_census_data(train_path, test_path)
    
    # Process training data
    train_clean = clean_data(train_df)
    train_engineered = engineer_features(train_clean)
    train_ready = prepare_features(train_engineered)
    
    # Process test data (same pipeline)
    test_clean = clean_data(test_df)
    test_engineered = engineer_features(test_clean)
    test_ready = prepare_features(test_engineered)
    
    # Align test features with training features
    train_features = [col for col in train_ready.columns if col != TARGET_COLUMN]
    test_ready = test_ready.reindex(columns=train_ready.columns, fill_value=0)
    
    # Split features and target
    X_train, y_train = split_features_target(train_ready)
    X_test, y_test = split_features_target(test_ready)
    
    # Train models
    lr_model = train_model(X_train, y_train, 'logistic')
    rf_model = train_model(X_train, y_train, 'random_forest')
    
    # Evaluate models
    lr_results = evaluate_model(lr_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    
    return {
        'logistic_regression': {
            'model': lr_model,
            'results': lr_results
        },
        'random_forest': {
            'model': rf_model,
            'results': rf_results
        },
        'data_shape': {
            'train': train_ready.shape,
            'test': test_ready.shape
        }
    }


# ========================================
# EVEN SIMPLER WITH FUNCTION COMPOSITION
# ========================================

from functools import reduce

def compose_pipeline(*functions):
    """Compose multiple functions into a pipeline"""
    return lambda x: reduce(lambda acc, f: f(acc), functions, x)

# Create preprocessing pipeline
preprocess_pipeline = compose_pipeline(
    clean_data,
    engineer_features,
    prepare_features
)

# Usage
def run_composed_pipeline(train_path: str, test_path: str):
    # Load data
    train_df, test_df = load_census_data(train_path, test_path)
    
    # Apply same preprocessing to both
    train_processed = preprocess_pipeline(train_df)
    test_processed = preprocess_pipeline(test_df)
    
    # Continue with modeling...
    return train_processed, test_processed


if __name__ == "__main__":
    # Example usage - much simpler!
    results = run_simple_pipeline(
        "Data/census_income_learn.csv",
        "Data/census_income_test.csv"
    )
    
    print("Logistic Regression Accuracy:", results['logistic_regression']['results']['accuracy'])
    print("Random Forest Accuracy:", results['random_forest']['results']['accuracy'])