"""
Car Price Prediction - Optimized Model Training for Deployment
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

print("="*60)
print("CAR PRICE PREDICTION - OPTIMIZED MODEL TRAINING")
print("="*60)

# Read Data
print("\n[1/7] Loading data...")
df = pd.read_csv("audi.csv")
print(f"Dataset shape: {df.shape}")

# Create X and Y
print("\n[2/7] Preparing features...")
X = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # All columns except price
Y = df.iloc[:, [2]].values  # price column

# Label Encoding for 'model' and 'fuelType' columns
print("\n[3/7] Encoding categorical variables...")
le1 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])  # model

le2 = LabelEncoder()
X[:, -4] = le2.fit_transform(X[:, -4])  # fuelType

# One Hot Encoding for 'transmission' column
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [2])],
    remainder='passthrough'
)
X = ct.fit_transform(X)

# Feature Scaling
print("\n[4/7] Scaling features...")
sc = StandardScaler()
X = sc.fit_transform(X)

# Train Test Split
print("\n[5/7] Splitting data...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)

# Dictionary to store model results
results = {}

# Model 1: Linear Regression (smallest model)
print("\n[6/7] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train.ravel())
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(Y_test, lr_pred)
lr_mae = mean_absolute_error(Y_test, lr_pred)
results['Linear Regression'] = {'model': lr_model, 'r2': lr_r2, 'mae': lr_mae}
print(f"Linear Regression - R2 Score: {lr_r2:.4f}, MAE: {lr_mae:.2f}")

# Model 2: Optimized Random Forest (smaller than default)
print("\nTraining Optimized Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=50,  # Reduced from 100
    max_depth=15,     # Limited depth
    min_samples_split=10,  # Increased to reduce overfitting
    min_samples_leaf=5,    # Increased to reduce model size
    random_state=0
)
rf_model.fit(X_train, Y_train.ravel())
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(Y_test, rf_pred)
rf_mae = mean_absolute_error(Y_test, rf_pred)
results['Optimized Random Forest'] = {'model': rf_model, 'r2': rf_r2, 'mae': rf_mae}
print(f"Optimized Random Forest - R2 Score: {rf_r2:.4f}, MAE: {rf_mae:.2f}")

# Find best model
print("\n[7/7] Model Comparison:")
print("-" * 60)
for name, result in results.items():
    print(f"{name:25s} - R2: {result['r2']:.4f}, MAE: {result['mae']:.2f}")
print("-" * 60)

best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]['r2']:.4f}")
print(f"MAE: {results[best_model_name]['mae']:.2f}")

# Save the best model and preprocessing objects
print("\nSaving optimized model and preprocessing objects...")

# Use protocol 4 for smaller file sizes
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f, protocol=4)

with open('label_encoder_model.pkl', 'wb') as f:
    pickle.dump(le1, f, protocol=4)

with open('label_encoder_fuel.pkl', 'wb') as f:
    pickle.dump(le2, f, protocol=4)

with open('column_transformer.pkl', 'wb') as f:
    pickle.dump(ct, f, protocol=4)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f, protocol=4)

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'r2_score': results[best_model_name]['r2'],
    'mae': results[best_model_name]['mae'],
    'all_results': {name: {'r2': r['r2'], 'mae': r['mae']} for name, r in results.items()}
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f, protocol=4)

print("\n" + "="*60)
print("Optimized training completed successfully!")
print("Files saved with reduced size:")
print("  - model.pkl (optimized)")
print("  - label_encoder_model.pkl")
print("  - label_encoder_fuel.pkl")
print("  - column_transformer.pkl")
print("  - scaler.pkl")
print("  - model_metadata.pkl")
print("="*60)