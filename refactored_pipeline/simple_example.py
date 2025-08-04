"""Simple example demonstrating the functional pipeline approach"""

from data_loader import load_and_preprocess_data, split_features_target
from model_trainer import train_logistic_regression, train_random_forest, evaluate_model
from visualization import plot_target_distribution, plot_feature_importance
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_simple_example():
    """Run a simple end-to-end example"""
    print("🚀 Running Simple Functional Pipeline Example")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("📊 Loading and preprocessing data...")
    train_df, test_df = load_and_preprocess_data()
    print(f"✅ Data loaded - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Step 2: Split features and target
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    
    # Step 3: Train models
    print("\n🤖 Training models...")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    print("✅ Models trained successfully")
    
    # Step 4: Evaluate models
    print("\n📈 Evaluating models...")
    lr_results = evaluate_model(lr_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"Logistic Regression - Accuracy: {lr_results['metrics']['accuracy']:.3f}, ROC AUC: {lr_results['metrics'].get('roc_auc', 'N/A'):.3f}")
    print(f"Random Forest      - Accuracy: {rf_results['metrics']['accuracy']:.3f}, ROC AUC: {rf_results['metrics'].get('roc_auc', 'N/A'):.3f}")
    
    # Step 5: Simple visualization (optional)
    try:
        print("\n📊 Generating basic visualization...")
        target_fig = plot_target_distribution(train_df)
        print("✅ Target distribution plot created")
        
        # You can open the plot in browser if needed
        # target_fig.show()
        
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")
    
    print("\n🎉 Simple pipeline completed!")
    return {
        'lr_accuracy': lr_results['metrics']['accuracy'],
        'rf_accuracy': rf_results['metrics']['accuracy'],
        'best_model': 'Random Forest' if rf_results['metrics']['accuracy'] > lr_results['metrics']['accuracy'] else 'Logistic Regression'
    }


def run_minimal_example():
    """Even simpler example - just the essentials"""
    print("⚡ Running Minimal Example")
    
    # Load data
    train_df, test_df = load_and_preprocess_data()
    
    # Split data
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    
    # Train and evaluate one model
    model = train_logistic_regression(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    
    print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    return results['metrics']['accuracy']


if __name__ == "__main__":
    # Run the simple example
    results = run_simple_example()
    print(f"\nBest Model: {results['best_model']}")
    
    print("\n" + "="*50)
    print("💡 Compare this to class-based approach:")
    print("   - No need to instantiate classes")
    print("   - Direct function calls")
    print("   - Easy to understand data flow")
    print("   - Simple to test individual functions")
    print("   - Composable and reusable")
    print("="*50)