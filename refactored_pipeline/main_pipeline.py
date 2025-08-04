"""Main pipeline script for Census Income Prediction"""

import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Import functional modules
from data_loader import load_and_preprocess_data, get_data_info
from model_trainer import run_complete_training_pipeline
from visualization import generate_eda_report, generate_model_visualizations
from config import RESULTS_DIR, PLOTS_DIR, TARGET_COLUMN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data_step() -> tuple:
    """Step 1: Load and preprocess data"""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
    logger.info("=" * 50)
    
    try:
        train_df, test_df = load_and_preprocess_data()
        
        data_info = get_data_info(train_df, test_df)
        
        logger.info(f"Data loaded successfully")
        logger.info(f"Train shape: {data_info['train_shape']}")
        logger.info(f"Test shape: {data_info['test_shape']}")
        logger.info(f"Feature count: {len([col for col in train_df.columns if col != TARGET_COLUMN])}")
        
        return train_df, test_df, data_info
        
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise


def run_eda_step(train_df, generate_visualizations: bool = True) -> Dict[str, Any]:
    """Step 2: Generate exploratory data analysis"""
    logger.info("=" * 50)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 50)
    
    if not generate_visualizations:
        logger.info("Skipping EDA visualization generation")
        return {}
        
    try:
        eda_figures = generate_eda_report(train_df, PLOTS_DIR)
        
        eda_results = {
            'visualizations_generated': len(eda_figures),
            'output_directory': str(PLOTS_DIR)
        }
        
        logger.info(f"Generated {len(eda_figures)} EDA visualizations")
        return eda_results
        
    except Exception as e:
        logger.error(f"Error in EDA generation: {e}")
        raise


def run_modeling_step(train_df, test_df) -> Dict[str, Any]:
    """Step 3: Train and evaluate models"""
    logger.info("=" * 50)
    logger.info("STEP 3: MODEL TRAINING AND EVALUATION")
    logger.info("=" * 50)
    
    try:
        training_results = run_complete_training_pipeline(train_df, test_df)
        
        # Extract key information
        best_model_name = training_results['best_model_name']
        final_metrics = training_results['final_test_metrics']
        comparison_df = training_results['comparison']
        
        modeling_results = {
            'best_model': best_model_name,
            'final_test_metrics': final_metrics,
            'models_trained': len(comparison_df),
            'model_comparison': comparison_df.to_dict('records')
        }
        
        logger.info(f"Trained {len(comparison_df)} models")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model accuracy: {final_metrics.get('accuracy', 'N/A'):.3f}")
        logger.info(f"Best model ROC AUC: {final_metrics.get('roc_auc', 'N/A'):.3f}")
        
        return training_results, modeling_results
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def run_model_visualization_step(training_results, generate_visualizations: bool = True) -> Dict[str, Any]:
    """Step 4: Generate model performance visualizations"""
    logger.info("=" * 50)
    logger.info("STEP 4: MODEL VISUALIZATION")
    logger.info("=" * 50)
    
    if not generate_visualizations:
        logger.info("Skipping model visualization generation")
        return {}
        
    try:
        models = training_results['models']
        evaluation_results = training_results['validation_results']
        comparison_df = training_results['comparison']
        best_model_name = training_results['best_model_name']
        feature_importance = training_results['feature_importance']
        
        # Generate visualizations
        viz_figures = generate_model_visualizations(
            models, evaluation_results, comparison_df, 
            best_model_name, feature_importance, PLOTS_DIR
        )
        
        model_viz_results = {
            'visualizations_generated': len(viz_figures),
            'best_model_analyzed': best_model_name,
            'feature_importance_top_features': len(feature_importance)
        }
        
        logger.info(f"Generated {len(viz_figures)} model performance visualizations")
        return model_viz_results
        
    except Exception as e:
        logger.error(f"Error in model visualization: {e}")
        raise


def save_pipeline_results(results: Dict[str, Any], timestamp: str) -> Path:
    """Save pipeline results to JSON file"""
    results_file = RESULTS_DIR / f"pipeline_results_{timestamp}.json"
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return obj
    
    # Prepare results for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            json_results[key] = convert_numpy(value)
    
    # Add metadata
    json_results['pipeline_metadata'] = {
        'timestamp': timestamp,
        'output_directories': {
            'plots': str(PLOTS_DIR),
            'results': str(RESULTS_DIR)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Pipeline results saved to {results_file}")
    return results_file


def run_complete_pipeline(generate_visualizations: bool = True) -> Dict[str, Any]:
    """Execute the complete census income prediction pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("🚀 Starting Census Income Prediction Pipeline")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Visualizations enabled: {generate_visualizations}")
    
    start_time = datetime.now()
    results = {}
    
    try:
        # Step 1: Data Loading
        train_df, test_df, data_info = load_data_step()
        results['data_info'] = data_info
        
        # Step 2: EDA
        eda_results = run_eda_step(train_df, generate_visualizations)
        results['eda'] = eda_results
        
        # Step 3: Model Training
        training_results, modeling_results = run_modeling_step(train_df, test_df)
        results['model_training'] = modeling_results
        
        # Step 4: Model Visualization
        viz_results = run_model_visualization_step(training_results, generate_visualizations)
        results['model_visualization'] = viz_results
        
        # Save results
        results_file = save_pipeline_results(results, timestamp)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Final summary
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration}")
        logger.info(f"Best Model: {results['model_training']['best_model']}")
        logger.info(f"Final Accuracy: {results['model_training']['final_test_metrics'].get('accuracy', 'N/A'):.3f}")
        logger.info(f"Results saved to: {results_file}")
        
        return {
            'success': True,
            'duration': str(duration),
            'results_file': str(results_file),
            'summary': results,
            'best_model': results['model_training']['best_model'],
            'best_model_path': training_results['model_saved_to']
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Pipeline execution stopped")
        return {
            'success': False,
            'error': str(e),
            'partial_results': results
        }


def run_simple_pipeline() -> Dict[str, Any]:
    """Run a simplified version of the pipeline for quick testing"""
    logger.info("🔥 Running Simplified Pipeline (No Visualizations)")
    
    try:
        # Load and preprocess data
        train_df, test_df = load_and_preprocess_data()
        logger.info(f"Data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Train models and get best one
        training_results = run_complete_training_pipeline(train_df, test_df)
        
        best_model_name = training_results['best_model_name']
        final_metrics = training_results['final_test_metrics']
        
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Accuracy: {final_metrics['accuracy']:.3f}")
        logger.info(f"ROC AUC: {final_metrics.get('roc_auc', 'N/A'):.3f}")
        
        return {
            'success': True,
            'best_model': best_model_name,
            'metrics': final_metrics,
            'model_path': training_results['model_saved_to']
        }
        
    except Exception as e:
        logger.error(f"Simple pipeline failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Census Income Prediction Pipeline")
    parser.add_argument(
        "--no-viz", 
        action="store_true", 
        help="Skip visualization generation for faster execution"
    )
    parser.add_argument(
        "--simple", 
        action="store_true", 
        help="Run simplified pipeline (no visualizations, minimal output)"
    )
    parser.add_argument(
        "--log-level", 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run appropriate pipeline
    if args.simple:
        results = run_simple_pipeline()
    else:
        results = run_complete_pipeline(generate_visualizations=not args.no_viz)
    
    # Print summary
    if results['success']:
        print("\n" + "="*50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        if 'best_model' in results:
            print(f"Best Model: {results['best_model']}")
        if 'metrics' in results:
            print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
        elif 'summary' in results:
            metrics = results['summary']['model_training']['final_test_metrics']
            print(f"Accuracy: {metrics['accuracy']:.3f}")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ PIPELINE FAILED!")
        print(f"Error: {results['error']}")
        print("="*50)
    
    # Exit with appropriate code
    exit_code = 0 if results['success'] else 1
    exit(exit_code)


if __name__ == "__main__":
    main()