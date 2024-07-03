import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import pickle
import argparse

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def fit(X, y):
    model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model

def run(train_path, model_path, scaler_path):
    df = load_data(train_path)
    X, y, scaler = preprocess_data(df)
    model = fit(X, y)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f'Model and scaler saved to {model_path} and {scaler_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_path', type=str, help='Path to the trainset file')
    parser.add_argument('--model_path', type=str, help='Path to save the trained model')
    parser.add_argument('--scaler_path', type=str, help='Path to save the trained scaler')
    
    args = parser.parse_args()
    run(args.train_path, args.model_path, args.scaler_path)
