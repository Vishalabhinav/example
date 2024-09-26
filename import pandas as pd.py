import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load product data
# Example: In your project, you would load real product data
data = pd.DataFrame({
    'Product_Name': ['A', 'B', 'C', 'D'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing'],
    'Price': [1000, 1500, 1200, 1300],
    'Discount': [100, 150, 120, 110],
    'Sales_Quantity': [50, 40, 30, 25]
})

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Category'], drop_first=True)

# Define features and target variable
X = data.drop(columns=['Price', 'Product_Name'])
y = data['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for regularized regression models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models and hyperparameters for GridSearch
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

param_grid = {
    'Ridge': {'alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'alpha': [0.1, 1.0, 10.0]},
    'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
}

best_model = None
best_score = -float('inf')
best_model_name = ''

# Train models and evaluate with R-squared score using GridSearchCV
for model_name in models:
    grid_search = GridSearchCV(models[model_name], param_grid[model_name], scoring='r2', cv=5)
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model from GridSearch
    best_estimator = grid_search.best_estimator_
    
    # Predict on the test data
    y_pred = best_estimator.predict(X_test_scaled)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} R-squared score: {r2}")
    
    # Select the best model based on R-squared
    if r2 > best_score:
        best_model = best_estimator
        best_score = r2
        best_model_name = model_name

# Output the best model
print(f"\nBest model is {best_model_name} with an R-squared score of {best_score}")

# Predict future prices using the best model
new_data = pd.DataFrame({
    'Discount': [80, 100],
    'Sales_Quantity': [35, 25],
    'Category_Clothing': [1, 0]  # Dummy variable for category
})

# Scale the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions for future prices
future_prices = best_model.predict(new_data_scaled)
print("\nPredicted Future Prices:", future_prices)
