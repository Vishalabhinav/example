import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(df):
    # Handle missing values (if any)
    df.fillna(df.mean(), inplace=True)
    X = df[['Product_Name', 'Category', 'Discount', 'Sales_Quantity']]
    y = df['Price']
    # Dummy encoding for categorical columns
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# Split and scale data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train model and evaluate
def train_and_evaluate(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

# Main function for AutoML
def automl_product_price_prediction(data_path):
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet()
    }

    best_model, best_score = None, -np.inf

    for model_name, model in models.items():
        score = train_and_evaluate(X_train, y_train, X_test, y_test, model)
        print(f"{model_name} R-squared score: {score}")
        if score > best_score:
            best_model, best_score = model, score

    print(f"Best model: {best_model.__class__.__name__} with R-squared score: {best_score}")

if __name__ == '__main__':
    automl_product_price_prediction('ecommerce_data.csv')
