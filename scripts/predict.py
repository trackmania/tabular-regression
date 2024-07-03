import pandas as pd
import pickle
import argparse

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, scaler):
    X = df
    X_scaled = scaler.transform(X)
    return X_scaled

def run(test_path, model_path, scaler_path, output_path):
    df = load_data(test_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    X_scaled = preprocess_data(df, scaler)
    predictions = model.predict(X_scaled)
    
    output_df = pd.DataFrame({'Predictions': predictions})
    output_df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference using a trained model')
    parser.add_argument('--test_path', type=str, help='Path to the testset file')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--scaler_path', type=str, help='Path to the scaler file')
    parser.add_argument('--output_path', type=str, help='Path to save the output')
    
    args = parser.parse_args()
    run(args.test_path, args.model_path, args.scaler_path, args.output_path)
