"""Visualization and analysis module for Census Income Prediction"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path

from config import PLOTS_DIR, TARGET_COLUMN, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def clean_feature_name(name: str) -> str:
    """Clean feature names for better display"""
    # Replace underscores with spaces and title case
    clean_name = name.replace('_', ' ').title()
    
    # Specific mappings for better readability
    mappings = {
        'Major Industry Code': 'Industry',
        'Major Occupation Code': 'Occupation',
        'Marital Stat': 'Marital Status',
        'Hispanic Origin': 'Hispanic Origin',
        'Class Of Worker': 'Employment Class',
        'Full Or Part Time Employment Stat': 'Employment Status',
        'Member Of A Labor Union': 'Union Member',
        'Tax Filer Stat': 'Tax Filing Status',
        'Own Business Or Self Employed': 'Self Employed',
        'Veterans Benefits': 'Veteran Benefits',
        'Fill Inc Questionnaire For Veterans Admin': 'Veteran Admin',
        'Enroll In Edu Inst Last Wk': 'Currently Enrolled'
    }
    
    return mappings.get(clean_name, clean_name)


def save_plot(fig: go.Figure, output_dir: Path, filename: str) -> Path:
    """Save a plotly figure to file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    fig.write_html(filepath)
    logger.info(f"Plot saved to {filepath}")
    
    return filepath


def plot_target_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> go.Figure:
    """Plot target variable distribution"""
    target_counts = df[TARGET_COLUMN].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['≤ $50K', '> $50K'],
            y=target_counts.values,
            text=[f'{count:,}<br>({count/len(df)*100:.1f}%)' for count in target_counts.values],
            textposition='auto',
            marker_color=['lightcoral', 'lightblue']
        )
    ])
    
    fig.update_layout(
        title='Income Level Distribution',
        xaxis_title='Income Level',
        yaxis_title='Count',
        showlegend=False
    )
    
    if save_path:
        save_plot(fig, save_path, 'target_distribution.html')
        
    return fig


def plot_continuous_distributions(df: pd.DataFrame, columns: List[str] = None, 
                                save_path: Optional[Path] = None) -> go.Figure:
    """Plot distributions of continuous features by target variable"""
    if columns is None:
        columns = [col for col in CONTINUOUS_FEATURES if col in df.columns]
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[clean_feature_name(col) for col in columns],
        vertical_spacing=0.08
    )
    
    colors = ['lightblue', 'lightcoral']
    target_labels = ['≤ $50K', '> $50K']
    
    for i, col in enumerate(columns):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        for target_val, color, label in zip([0, 1], colors, target_labels):
            subset = df[df[TARGET_COLUMN] == target_val][col].dropna()
            
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=label,
                    opacity=0.7,
                    marker_color=color,
                    showlegend=(i == 0)  # Only show legend for first subplot
                ),
                row=row, col=col_pos
            )
    
    fig.update_layout(
        title='Distribution of Continuous Features by Income Level',
        height=300 * n_rows,
        barmode='overlay'
    )
    
    if save_path:
        save_plot(fig, save_path, 'continuous_distributions.html')
        
    return fig


def plot_categorical_distributions(df: pd.DataFrame, columns: List[str] = None,
                                 max_categories: int = 10, save_path: Optional[Path] = None) -> go.Figure:
    """Plot distributions of categorical features by target variable"""
    if columns is None:
        # Select a subset of important categorical features
        important_cats = ['education', 'major_occupation_code', 'major_industry_code', 
                        'marital_stat', 'sex', 'race']
        columns = [col for col in important_cats if col in df.columns]
    
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[clean_feature_name(col) for col in columns],
        vertical_spacing=0.12
    )
    
    for i, col in enumerate(columns):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        # Get top categories
        top_cats = df[col].value_counts().head(max_categories).index
        subset_df = df[df[col].isin(top_cats)]
        
        # Calculate proportions
        cross_tab = pd.crosstab(subset_df[col], subset_df[TARGET_COLUMN], normalize='index')
        
        fig.add_trace(
            go.Bar(
                x=cross_tab.index,
                y=cross_tab[0],
                name='≤ $50K',
                marker_color='lightblue',
                showlegend=(i == 0)
            ),
            row=row, col=col_pos
        )
        
        fig.add_trace(
            go.Bar(
                x=cross_tab.index,
                y=cross_tab[1],
                name='> $50K',
                marker_color='lightcoral',
                showlegend=(i == 0)
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title='Categorical Feature Distributions by Income Level',
        height=400 * n_rows,
        barmode='stack'
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    if save_path:
        save_plot(fig, save_path, 'categorical_distributions.html')
        
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None) -> go.Figure:
    """Plot correlation heatmap for continuous features"""
    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in df.columns]
    corr_matrix = df[continuous_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[clean_feature_name(col) for col in corr_matrix.columns],
        y=[clean_feature_name(col) for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Matrix - Continuous Features',
        width=600,
        height=500
    )
    
    if save_path:
        save_plot(fig, save_path, 'correlation_heatmap.html')
        
    return fig


def plot_age_income_analysis(df: pd.DataFrame, save_path: Optional[Path] = None) -> go.Figure:
    """Analyze age vs income relationship"""
    # Create age groups
    df_copy = df.copy()
    df_copy['age_group'] = pd.cut(
        df_copy['age'], 
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    # Calculate income rates by age group
    age_income = df_copy.groupby('age_group')[TARGET_COLUMN].agg(['count', 'sum', 'mean']).reset_index()
    age_income['high_income_rate'] = age_income['mean'] * 100
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Sample Size by Age Group', 'High Income Rate by Age Group'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sample size
    fig.add_trace(
        go.Bar(
            x=age_income['age_group'],
            y=age_income['count'],
            name='Sample Size',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # High income rate
    fig.add_trace(
        go.Bar(
            x=age_income['age_group'],
            y=age_income['high_income_rate'],
            name='High Income Rate (%)',
            marker_color='lightcoral'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Age Group Analysis',
        showlegend=False,
        height=400
    )
    
    if save_path:
        save_plot(fig, save_path, 'age_income_analysis.html')
        
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str,
                          top_n: int = 15, save_path: Optional[Path] = None) -> go.Figure:
    """Plot feature importance"""
    top_features = importance_df.head(top_n)
    
    # Determine importance column
    importance_col = 'importance' if 'importance' in top_features.columns else 'abs_coefficient'
    
    fig = go.Figure(data=go.Bar(
        x=top_features[importance_col],
        y=top_features['feature'],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    if save_path:
        filename = f'{model_name}_feature_importance.html'
        save_plot(fig, save_path, filename)
        
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Optional[Path] = None) -> go.Figure:
    """Plot model performance comparison"""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    fig = go.Figure()
    
    for metric in available_metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=comparison_df['model'],
            y=comparison_df[metric],
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    if save_path:
        save_plot(fig, save_path, 'model_comparison.html')
        
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Optional[Path] = None) -> go.Figure:
    """Plot confusion matrix"""
    labels = ['≤ $50K', '> $50K']
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f'{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)',
                    showarrow=False,
                    font=dict(color="white" if cm[i,j] > cm.max()/2 else "black")
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name.replace("_", " ").title()}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        annotations=annotations,
        width=400,
        height=400
    )
    
    if save_path:
        filename = f'{model_name}_confusion_matrix.html'
        save_plot(fig, save_path, filename)
        
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                  model_name: str, save_path: Optional[Path] = None) -> go.Figure:
    """Plot ROC curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name.replace("_", " ").title()}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=500,
        height=500
    )
    
    if save_path:
        filename = f'{model_name}_roc_curve.html'
        save_plot(fig, save_path, filename)
        
    return fig


def generate_eda_report(df: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Dict[str, go.Figure]:
    """Generate comprehensive EDA report"""
    logger.info("Generating comprehensive EDA report...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    figures = {}
    
    # Target distribution
    figures['target_distribution'] = plot_target_distribution(df, output_dir)
    
    # Continuous features
    figures['continuous_distributions'] = plot_continuous_distributions(df, save_path=output_dir)
    
    # Categorical features
    figures['categorical_distributions'] = plot_categorical_distributions(df, save_path=output_dir)
    
    # Correlation heatmap
    figures['correlation_heatmap'] = plot_correlation_heatmap(df, output_dir)
    
    # Age analysis
    figures['age_income_analysis'] = plot_age_income_analysis(df, output_dir)
    
    logger.info(f"EDA report generated with {len(figures)} visualizations")
    return figures


def generate_model_visualizations(models: Dict[str, Any], evaluation_results: Dict[str, Dict],
                                comparison_df: pd.DataFrame, best_model_name: str,
                                feature_importance: pd.DataFrame, output_dir: Path = PLOTS_DIR) -> Dict[str, go.Figure]:
    """Generate model performance visualizations"""
    logger.info("Generating model performance visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    figures = {}
    
    # Model comparison
    figures['model_comparison'] = plot_model_comparison(comparison_df, output_dir)
    
    # Feature importance for best model
    figures['feature_importance'] = plot_feature_importance(feature_importance, best_model_name, save_path=output_dir)
    
    # Confusion matrix for best model
    if best_model_name in evaluation_results:
        cm = evaluation_results[best_model_name]['confusion_matrix']
        figures['confusion_matrix'] = plot_confusion_matrix(cm, best_model_name, output_dir)
    
    logger.info(f"Model visualizations generated with {len(figures)} charts")
    return figures


if __name__ == "__main__":
    # Example usage
    from data_loader import load_and_preprocess_data
    
    try:
        train_df, test_df = load_and_preprocess_data()
        
        # Generate EDA report
        figures = generate_eda_report(train_df)
        print(f"Generated {len(figures)} visualizations")
        
    except Exception as e:
        logger.error(f"Error in visualization: {e}")