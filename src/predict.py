import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sys
#from  models.regression_model1 import LassoRegression 
#from  models.regression_model2 import RidgeRegression
from data_preprocessing import Preprocessor


def load_model(model_path):
    #Load trained model from file
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    
#Define Regression Metrics to evaluate model
def evaluate_model(y_val, y_pred):
    mse = np.mean((y_val-y_pred)**2) #mean squared error
    rmse = np.sqrt(mse) #root mean square error/validation error
    #find r^2 score
    #use formula: 1 - [(Residual Sum of Squares (RSS))/(Total Sum of Squares(TSS))]
    rss = np.sum((y_val-y_pred)**2)
    mean= np.mean(y_val)
    tss = np.sum((y_val-mean)**2)
    r_sqaured_score= 1-(rss/tss)

    return {'mse': mse, 'rmse': rmse, 'r_squared_score': r_sqaured_score}

def write_metrics_file(metrics, filepath):
    #write in req format
    with open(filepath, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}\n")
        f.write(f"R-squared (R²) Score: {metrics['r_sqaured_score']:.2f}\n")

def save_predictions(predictions, output_path):
    """Save predictions to CSV file"""
    np.savetxt(output_path, predictions, delimiter=',',fmt='%.6f')

def main():
    parser = argparse.ArgumentParser(description='Predict using trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path to save metrics')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path to save predictions')
    
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    try:
    # Load preprocessor and model
        model = load_model(args.model_path)
        print(f"Model loaded from: {args.model_path}")

        preprocessor_path = os.path.join(os.path.dirname(args.model_path),'preprocessor.pkl')
        
        if not os.path.exists(preprocessor_path):
            print(f"Error: Preprocessor file not found at {preprocessor_path}")
            return
        preprocessor = load_model(preprocessor_path)
        print('Preprocessor loeaded successfully')

        #Load and process data
        df = pd.read_csv(args.data_path)
        print(f'Data loaded: {df.shape}')

        #Transform data
        X,y=preprocessor.transform(df, 'Price')
        print(f'Data preprocessed. Shape{X.shape}')

         # Make predictions
        predictions = model.predict(X)
        print(f'Predictions made. Shape: {predictions.shape}')
    
    # Calculate metrics if target is available
        if y is not None:
            metrics = evaluate_model(y, predictions)
            
            # Create results directory if it doesn't exist
            os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
            write_metrics_file(metrics, args.metrics_output_path)
            print(f"Metrics saved to {args.metrics_output_path}")
        
        # Save predictions
        os.makedirs(os.path.dirname(args.predictions_output_path), exist_ok=True)
        save_predictions(predictions, args.predictions_output_path)
        print(f"Predictions saved to {args.predictions_output_path}")
        
        print("Prediction completed successfully!")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise    
   

if __name__ == "__main__":
    main()