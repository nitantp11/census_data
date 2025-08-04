"""Utility functions for Census Income Prediction Pipeline"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import joblib

logger = logging.getLogger(__name__)


# ==============================================
# DATA QUALITY FUNCTIONS
# ==============================================

def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for missing values in dataset"""
    missing_stats = {}
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        missing_stats[col] = {
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2),
            'data_type': str(df[col].dtype)
        }
    
    # Summary
    total_missing = sum(stats['missing_count'] for stats in missing_stats.values())
    total_cells = len(df) * len(df.columns)
    
    summary = {
        'total_missing_values': total_missing,
        'total_cells': total_cells,
        'overall_missing_percentage': round((total_missing / total_cells) * 100, 2),
        'columns_with_missing': len([col for col, stats in missing_stats.items() 
                                   if stats['missing_count'] > 0]),
        'details': missing_stats
    }
    
    return summary


def check_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data types and suggest improvements"""
    type_summary = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        
        type_summary[col] = {
            'current_dtype': str(dtype),
            'unique_values': int(unique_count),
            'sample_values': df[col].dropna().head(3).tolist(),
            'memory_usage_mb': round(df[col].memory_usage(deep=True) / 1024 / 1024, 2)
        }
        
        # Suggest optimizations
        if dtype == 'object' and unique_count < len(df) * 0.1:
            type_summary[col]['suggestion'] = 'Consider categorical dtype'
        elif dtype in ['int64', 'float64']:
            if df[col].min() >= 0 and df[col].max() < 255:
                type_summary[col]['suggestion'] = 'Consider uint8'
            elif df[col].min() >= -128 and df[col].max() < 127:
                type_summary[col]['suggestion'] = 'Consider int8'
    
    return type_summary


def detect_outliers(df: pd.DataFrame, columns: List[str] = None, 
                   method: str = 'iqr') -> Dict[str, Any]:
    """Detect outliers in numerical columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_summary = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3]
            lower_bound = None
            upper_bound = None
        
        outlier_summary[col] = {
            'outlier_count': len(outliers),
            'outlier_percentage': round((len(outliers) / len(series)) * 100, 2),
            'outlier_values': outliers.head(5).tolist() if len(outliers) > 0 else [],
            'bounds': {
                'lower': float(lower_bound) if lower_bound is not None else None,
                'upper': float(upper_bound) if upper_bound is not None else None
            }
        }
    
    return outlier_summary


# ==============================================
# FEATURE ANALYSIS FUNCTIONS
# ==============================================

def analyze_feature_importance(importance_df: pd.DataFrame, 
                              top_n: int = 20) -> Dict[str, Any]:
    """Analyze and categorize feature importance"""
    if importance_df.empty:
        return {}
    
    # Determine importance column
    importance_col = 'importance' if 'importance' in importance_df.columns else 'abs_coefficient'
    
    top_features = importance_df.head(top_n)
    
    # Categorize features
    categories = {
        'demographic': ['age', 'sex', 'race', 'hispanic', 'marital'],
        'education': ['education', 'school', 'degree'],
        'employment': ['occupation', 'industry', 'worker', 'employment', 'union', 'business'],
        'financial': ['capital', 'dividend', 'wage', 'income'],
        'geographic': ['region', 'state', 'residence'],
        'household': ['household', 'family', 'member']
    }
    
    categorized_features = {cat: [] for cat in categories.keys()}
    categorized_features['other'] = []
    
    for _, row in top_features.iterrows():
        feature_name = row['feature'].lower()
        categorized = False
        
        for category, keywords in categories.items():
            if any(keyword in feature_name for keyword in keywords):
                categorized_features[category].append({
                    'feature': row['feature'],
                    'importance': float(row[importance_col])
                })
                categorized = True
                break
        
        if not categorized:
            categorized_features['other'].append({
                'feature': row['feature'],
                'importance': float(row[importance_col])
            })
    
    # Summary statistics
    total_importance = top_features[importance_col].sum()
    
    analysis = {
        'top_features_count': len(top_features),
        'total_importance_captured': float(total_importance),
        'categories': categorized_features,
        'category_importance': {
            cat: sum(feat['importance'] for feat in features) 
            for cat, features in categorized_features.items()
        }
    }
    
    return analysis


def analyze_categorical_features(df: pd.DataFrame, target_col: str,
                                min_frequency: int = 10) -> Dict[str, Any]:
    """Analyze categorical features and their relationship with target"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    analysis = {}
    
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        
        # Filter out low-frequency categories
        frequent_values = value_counts[value_counts >= min_frequency]
        
        # Calculate target rates for each category
        target_rates = df.groupby(col)[target_col].mean()
        
        analysis[col] = {
            'unique_values': int(df[col].nunique()),
            'most_frequent_value': str(value_counts.index[0]),
            'most_frequent_count': int(value_counts.iloc[0]),
            'frequent_categories': len(frequent_values),
            'target_rate_range': {
                'min': float(target_rates.min()),
                'max': float(target_rates.max()),
                'std': float(target_rates.std())
            },
            'high_target_categories': target_rates.nlargest(3).to_dict(),
            'low_target_categories': target_rates.nsmallest(3).to_dict()
        }
    
    return analysis


# ==============================================
# MODEL EVALUATION FUNCTIONS
# ==============================================

def calculate_business_metrics(confusion_matrix: np.ndarray, 
                              cost_fp: float = 1.0, cost_fn: float = 5.0) -> Dict[str, float]:
    """Calculate business-oriented metrics"""
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    # Basic rates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Business metrics
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    cost_per_prediction = total_cost / (tn + fp + fn + tp)
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'total_cost': total_cost,
        'cost_per_prediction': cost_per_prediction,
        'cost_savings_vs_all_positive': total_cost - ((tn + fp + fn + tp) * cost_fn)
    }


def compare_model_performance(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Compare performance across multiple models"""
    comparison_data = []
    
    for model_name, results in results_dict.items():
        if 'metrics' in results:
            row = {'model': model_name}
            row.update(results['metrics'])
            comparison_data.append(row)
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add rankings
    metrics_to_rank = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics_to_rank:
        if metric in comparison_df.columns:
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
    
    # Calculate average rank
    rank_cols = [col for col in comparison_df.columns if col.endswith('_rank')]
    if rank_cols:
        comparison_df['average_rank'] = comparison_df[rank_cols].mean(axis=1)
        comparison_df = comparison_df.sort_values('average_rank')
    
    return comparison_df


# ==============================================
# FILE MANAGEMENT FUNCTIONS
# ==============================================

def save_results(results: Dict[str, Any], filepath: Path, 
                format: str = 'json') -> Path:
    """Save results in specified format"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            joblib.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {filepath}")
    return filepath


def load_results(filepath: Path, format: str = 'json') -> Dict[str, Any]:
    """Load results from file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        return joblib.load(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_project_structure(base_dir: Path) -> Dict[str, Path]:
    """Create standard project directory structure"""
    directories = {
        'data': base_dir / 'Data',
        'models': base_dir / 'models',
        'plots': base_dir / 'plots',
        'results': base_dir / 'results',
        'logs': base_dir / 'logs'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return directories


# ==============================================
# LOGGING FUNCTIONS
# ==============================================

def setup_logging(log_file: str = 'pipeline.log', level: str = 'INFO') -> None:
    """Setup comprehensive logging configuration"""
    log_level = getattr(logging, level.upper())
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def run_data_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Run comprehensive data quality assessment"""
    logger.info("Running data quality assessment...")
    
    quality_report = {
        'missing_values': check_missing_values(df),
        'data_types': check_data_types(df),
        'outliers': detect_outliers(df),
        'shape': df.shape,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }
    
    logger.info("Data quality assessment completed")
    return quality_report


def analyze_dataset_summary(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """Generate comprehensive dataset summary"""
    summary = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict()
        },
        'missing_data': check_missing_values(df),
        'numerical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
    }
    
    if target_col and target_col in df.columns:
        summary['target_analysis'] = {
            'distribution': df[target_col].value_counts().to_dict(),
            'missing_count': int(df[target_col].isnull().sum())
        }
        
        # Categorical feature analysis if target is provided
        summary['categorical_analysis'] = analyze_categorical_features(df, target_col)
    
    return summary


if __name__ == "__main__":
    # Example usage of utility functions
    print("Census Income Prediction - Utility Functions")
    print("Available functions:")
    print("- check_missing_values()")
    print("- check_data_types()")
    print("- detect_outliers()")
    print("- analyze_feature_importance()")
    print("- calculate_business_metrics()")
    print("- run_data_quality_check()")
    print("- analyze_dataset_summary()")