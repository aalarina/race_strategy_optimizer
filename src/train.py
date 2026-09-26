import os 
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
data_path = './data/f1_clean_data_for_ml.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The dataset file '{data_path}' does not exist. Please check the path and try again.")

df = pd.read_csv(data_path)

# Preprocessing the dataset: convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['GrandPrix', 'Driver', 'Compound'], drop_first=True)

# Split the dataset into features and target variable
X = df_encoded.drop('LapTime_Delta', axis=1)
if 'LapTime_Seconds' in X.columns:
    X = X.drop('LapTime_Seconds', axis=1)

y = df_encoded['LapTime_Delta']

# Save the structure of the dataset for future reference
model_features = X.columns.tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
# 150 trees and low learning rate to prevent overfitting
model = XGBRegressor(
    n_estimators=150, 
    learning_rate=0.03,
    max_depth=4, # Optimal depth 
    subsample=0.8, # Using 80% of the data 
    colsample_bytree=0.8, # Using 80% of features 
    reg_alpha=0.1, #L1 regularization
    reg_lambda=1.0, #L2 regularization 
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"Training completed. Model performance on the test set:")
print(f"Mean Absolute Error (MAE): {mae:.3f} seconds")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} seconds")

# Save the trained model and the feature structure for future use
os.makedirs('./models', exist_ok=True)
with open('./models/f1_tyre_degradation_model.pkl', 'wb') as f:
    pickle.dump((model, model_features), f)

print("Model and feature structure saved to './models/f1_tyre_degradation_model.pkl'.")