import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import pandas as pd
import numpy as np
from data_preprocessing import Preprocessor
from train_model import LassoRegression, RidgeRegression

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    with open('models/regression_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor

def standardize_features(X, mean=None, std=None):
    """Standardize features to have mean 0 and std 1"""
    X = np.array(X, dtype=np.float64)
    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    
    # Avoid division by zero for constant columns
    std_replaced = np.where(std == 0, 1, std)
    X_scaled = (X - mean) / std_replaced
    return X_scaled, mean, std_replaced

def generate_predictions():
    """Generate predictions for the training data"""
    print("Loading data...")
    df = pd.read_csv('data/Laptop Price - Laptop Price.csv')
    
    print("Loading model and preprocessor...")
    model, preprocessor = load_model_and_preprocessor()
    
    print("Preprocessing data...")
    X, y = preprocessor.fit_transform(df, 'Price')
    
    print("Standardizing features...")
    X_scaled, mean, std = standardize_features(X)
    
    print("Generating predictions...")
    predictions = model.predict(X_scaled)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual_Price': y.values,
        'Predicted_Price': predictions,
        'Residual': y.values - predictions,
        'Absolute_Error': np.abs(y.values - predictions),
        'Percentage_Error': np.abs((y.values - predictions) / y.values) * 100
    })
    
    # Add some sample features for context
    results_df['Company'] = df['Company'].values
    results_df['TypeName'] = df['TypeName'].values
    results_df['RAM'] = df['Ram'].values
    
    # Save predictions
    results_df.to_csv('results/train_predictions.csv', index=False)
    print(f"Predictions saved to results/train_predictions.csv")
    
    # Print some statistics
    mse = np.mean((y.values - predictions)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y.values - predictions))
    r2 = 1 - (np.sum((y.values - predictions)**2) / np.sum((y.values - np.mean(y.values))**2))
    
    print(f"\nPrediction Statistics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Percentage Error: {results_df['Percentage_Error'].mean():.2f}%")
    
    return results_df

if __name__ == "__main__":
    results = generate_predictions()
    print("\nFirst 10 predictions:")
    print(results[['Company', 'TypeName', 'RAM', 'Actual_Price', 'Predicted_Price', 'Percentage_Error']].head(10))
