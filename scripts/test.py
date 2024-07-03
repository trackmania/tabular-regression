from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
import pandas as pd

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y, model_type='random_forest'):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=200, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'svm':
        model = SVR()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    print(f'{model_type} Validation RMSE: {rmse}')
    print(f'Feature importances: {model.feature_importances_}')
    
    return y_pred, y_val

df = pd.read_csv("/home/alexey/test_task_2/data/train.csv")
X, y, scaler = preprocess_data(df)
pred, val = train_model(X, y, model_type="extra_trees")