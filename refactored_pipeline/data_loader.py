"""Data loading and preprocessing module for Census Income Prediction"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, COLUMN_NAMES, TARGET_COLUMN,
    FEATURES_TO_DROP, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_census_data(train_path: str = None, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from CSV files"""
    train_path = train_path or TRAIN_DATA_PATH
    test_path = test_path or TEST_DATA_PATH
    
    try:
        logger.info("Loading training data...")
        train_df = pd.read_csv(train_path, header=None)
        train_df.columns = COLUMN_NAMES
        
        logger.info("Loading test data...")
        test_df = pd.read_csv(test_path, header=None)
        test_df.columns = COLUMN_NAMES
        
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def get_data_info(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic information about the datasets"""
    info = {
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'train_missing': train_df.isnull().sum().sum(),
        'test_missing': test_df.isnull().sum().sum(),
        'target_distribution': train_df[TARGET_COLUMN].value_counts().to_dict(),
        'continuous_features': CONTINUOUS_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'features_to_drop': FEATURES_TO_DROP
    }
    
    return info


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Validate data integrity"""
    # Check for required columns
    required_cols = set(COLUMN_NAMES)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    if not required_cols.issubset(train_cols):
        missing_train = required_cols - train_cols
        logger.error(f"Missing columns in train data: {missing_train}")
        return False
        
    if not required_cols.issubset(test_cols):
        missing_test = required_cols - test_cols
        logger.error(f"Missing columns in test data: {missing_test}")
        return False
        
    # Check target column
    if TARGET_COLUMN not in train_df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found")
        return False
        
    logger.info("Data validation passed")
    return True

def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe by handling missing values and data types"""
    df_clean = df.copy()
    
    # Handle missing values (represented as '?' in this dataset)
    logger.info("Handling missing values...")
    df_clean = df_clean.replace('?', np.nan)
    
    # Convert continuous features to numeric
    for col in CONTINUOUS_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean categorical features (strip whitespace)
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Handle target variable
    if TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].str.strip()
        # Convert to binary (assuming '50000+' means high income)
        df_clean[TARGET_COLUMN] = (df_clean[TARGET_COLUMN] == '50000+').astype(int)
    
    logger.info(f"Data cleaned. Shape: {df_clean.shape}")
    return df_clean


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features"""
    df_eng = df.copy()
    
    # Age groups
    if 'age' in df_eng.columns:
        df_eng['age_group'] = pd.cut(
            df_eng['age'], 
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
    
    # Binary indicators for financial features
    if 'capital_gains' in df_eng.columns:
        df_eng['has_capital_gain'] = (df_eng['capital_gains'] > 0).astype(int)
        
    if 'capital_losses' in df_eng.columns:
        df_eng['has_capital_loss'] = (df_eng['capital_losses'] > 0).astype(int)
        
    if 'dividends_from_stocks' in df_eng.columns:
        df_eng['has_dividends'] = (df_eng['dividends_from_stocks'] > 0).astype(int)
    
    # Work intensity
    if 'weeks_worked_in_year' in df_eng.columns:
        df_eng['work_intensity'] = pd.cut(
            df_eng['weeks_worked_in_year'],
            bins=[0, 26, 39, 52],
            labels=['part_year', 'most_year', 'full_year']
        )
    
    logger.info("Feature engineering completed")
    return df_eng


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling"""
    df_prep = df.copy()
    
    # Drop irrelevant features
    drop_cols = [col for col in FEATURES_TO_DROP if col in df_prep.columns]
    df_prep = df_prep.drop(columns=drop_cols)
    logger.info(f"Dropped {len(drop_cols)} irrelevant columns")
    
    # Handle missing values
    # For continuous features, fill with median
    for col in CONTINUOUS_FEATURES:
        if col in df_prep.columns:
            median_val = df_prep[col].median()
            df_prep[col] = df_prep[col].fillna(median_val)
    
    # For categorical features, fill with mode or 'Unknown'
    categorical_cols = df_prep.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
    
    for col in categorical_cols:
        if df_prep[col].isnull().sum() > 0:
            mode_val = df_prep[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_prep[col] = df_prep[col].fillna(fill_val)
    
    # Remove original columns that were engineered
    to_drop = []
    if 'age_group' in df_prep.columns and 'age' in df_prep.columns:
        to_drop.append('age')
    if 'work_intensity' in df_prep.columns and 'weeks_worked_in_year' in df_prep.columns:
        to_drop.append('weeks_worked_in_year')
        
    df_prep = df_prep.drop(columns=[col for col in to_drop if col in df_prep.columns])
    
    # One-hot encode categorical variables
    categorical_cols = df_prep.select_dtypes(include=['object', 'category']).columns
    if TARGET_COLUMN in categorical_cols:
        categorical_cols = categorical_cols.drop(TARGET_COLUMN)
    
    if len(categorical_cols) > 0:
        df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)
        logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns")
    
    feature_columns = [col for col in df_prep.columns if col != TARGET_COLUMN]
    logger.info(f"Final feature count: {len(feature_columns)}")
    
    return df_prep


def preprocess_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Complete preprocessing pipeline"""
    logger.info("Starting preprocessing pipeline...")
    
    # Clean data
    train_clean = clean_raw_dataframe(train_df)
    test_clean = clean_raw_dataframe(test_df)
    
    # Feature engineering
    train_eng = create_engineered_features(train_clean)
    test_eng = create_engineered_features(test_clean)
    
    # Prepare features
    train_prep = prepare_features(train_eng)
    test_prep = prepare_features(test_eng)
    
    # Ensure test set has same columns as train set
    test_prep = test_prep.reindex(columns=train_prep.columns, fill_value=0)
    
    logger.info("Preprocessing pipeline completed")
    logger.info(f"Train shape: {train_prep.shape}, Test shape: {test_prep.shape}")
    
    return train_prep, test_prep


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target"""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to load and preprocess data in one step"""
    # Load data
    train_df, test_df = load_census_data()
    
    # Validate data
    if not validate_data(train_df, test_df):
        raise ValueError("Data validation failed")
    
    # Preprocess data
    train_processed, test_processed = preprocess_pipeline(train_df, test_df)
    
    return train_processed, test_processed


if __name__ == "__main__":
    # Example usage
    try:
        train_df, test_df = load_and_preprocess_data()
        print(f"Processed data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
        print(f"Target distribution:\n{train_df[TARGET_COLUMN].value_counts()}")
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
