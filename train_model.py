"""
Car Price Prediction - Model Training 
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

print("="*60)
print("CAR PRICE PREDICTION - MODEL TRAINING")
print("="*60)

# Read Data
print("\n[1/9] Loading data...")
df = pd.read_csv("audi.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Display basic info
print("\n[2/9] Dataset overview:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values:")
print(df.isna().sum())
print("\nBasic statistics:")
print(df.describe())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Create X and Y
print("\n[3/9] Preparing features...")
X = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values  # All columns except price
Y = df.iloc[:, [2]].values  # price column

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

# Label Encoding for 'model' and 'fuelType' columns
print("\n[4/9] Encoding categorical variables...")
le1 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])  # model

le2 = LabelEncoder()
X[:, -4] = le2.fit_transform(X[:, -4])  # fuelType

# One Hot Encoding for 'transmission' column (index 2 after label encoding)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [2])],
    remainder='passthrough'
)
X = ct.fit_transform(X)
print(f"X shape after encoding: {X.shape}")

# Feature Scaling - Standardization
print("\n[5/9] Scaling features...")
sc = StandardScaler()
X = sc.fit_transform(X)

# Train Test Split
print("\n[6/9] Splitting data...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Dictionary to store model results
results = {}

# Model 1: Random Forest Regressor
print("\n[7/9] Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, Y_train.ravel())
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(Y_test, rf_pred)
rf_mae = mean_absolute_error(Y_test, rf_pred)
results['Random Forest'] = {'model': rf_model, 'r2': rf_r2, 'mae': rf_mae}
print(f"Random Forest - R2 Score: {rf_r2:.4f}, MAE: {rf_mae:.2f}")

# Model 2: Linear Regression
print("\n[8/9] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train.ravel())
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(Y_test, lr_pred)
lr_mae = mean_absolute_error(Y_test, lr_pred)
results['Linear Regression'] = {'model': lr_model, 'r2': lr_r2, 'mae': lr_mae}
print(f"Linear Regression - R2 Score: {lr_r2:.4f}, MAE: {lr_mae:.2f}")

# Model 3: Extra Trees Regressor
print("\nTraining Extra Trees Regressor...")
et_model = ExtraTreesRegressor(n_estimators=120, random_state=0)
et_model.fit(X_train, Y_train.ravel())
et_pred = et_model.predict(X_test)
et_r2 = r2_score(Y_test, et_pred)
et_mae = mean_absolute_error(Y_test, et_pred)
results['Extra Trees'] = {'model': et_model, 'r2': et_r2, 'mae': et_mae}
print(f"Extra Trees - R2 Score: {et_r2:.4f}, MAE: {et_mae:.2f}")

# Model 4: CatBoost Regressor (if available)
if CATBOOST_AVAILABLE:
    print("\nTraining CatBoost Regressor...")
    cat_model = CatBoostRegressor(iterations=500, learning_rate=0.1, 
                                   depth=6, verbose=False, random_state=0)
    cat_model.fit(X_train, Y_train.ravel())
    cat_pred = cat_model.predict(X_test)
    cat_r2 = r2_score(Y_test, cat_pred)
    cat_mae = mean_absolute_error(Y_test, cat_pred)
    results['CatBoost'] = {'model': cat_model, 'r2': cat_r2, 'mae': cat_mae}
    print(f"CatBoost - R2 Score: {cat_r2:.4f}, MAE: {cat_mae:.2f}")

# Find best model
print("\n[9/9] Model Comparison:")
print("-" * 60)
for name, result in results.items():
    print(f"{name:20s} - R2: {result['r2']:.4f}, MAE: {result['mae']:.2f}")
print("-" * 60)

best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]['r2']:.4f}")
print(f"MAE: {results[best_model_name]['mae']:.2f}")

# Save the best model and preprocessing objects
print("\nSaving model and preprocessing objects...")
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('label_encoder_model.pkl', 'wb') as f:
    pickle.dump(le1, f)

with open('label_encoder_fuel.pkl', 'wb') as f:
    pickle.dump(le2, f)

with open('column_transformer.pkl', 'wb') as f:
    pickle.dump(ct, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'r2_score': results[best_model_name]['r2'],
    'mae': results[best_model_name]['mae'],
    'all_results': {name: {'r2': r['r2'], 'mae': r['mae']} for name, r in results.items()}
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "="*60)
print("Training completed successfully!")
print("Files saved:")
print("  - model.pkl")
print("  - label_encoder_model.pkl")
print("  - label_encoder_fuel.pkl")
print("  - column_transformer.pkl")
print("  - scaler.pkl")
print("  - model_metadata.pkl")
print("="*60)
