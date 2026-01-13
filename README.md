# 🚗 Audi Car Price Prediction System

A complete machine learning application for predicting Audi car prices using various regression models and deployed with Streamlit.

## 📋 Features

- **Multiple ML Models**: Trains and compares Random Forest, Linear Regression, Extra Trees, and CatBoost
- **Interactive Web Interface**: Beautiful Streamlit dashboard for predictions
- **Data Visualization**: Comprehensive charts and insights
- **Model Performance Tracking**: Compare different models and their metrics
- **Real-time Predictions**: Instant price predictions based on car features

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train multiple regression models
- Compare their performance
- Save the best model and preprocessing objects

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`



## 🎯 How to Use

### Training Models

The `train_model.py` script will:
1. Load the Audi car dataset
2. Perform exploratory data analysis
3. Preprocess data (encoding, scaling)
4. Train multiple models:
   - Random Forest Regressor
   - Linear Regression
   - Extra Trees Regressor
5. Compare models and save the best one

### Making Predictions

Using the Streamlit app:
1. Navigate to the "Predict Price" tab
2. Enter car details:
   - Model (e.g., A1, A3, A4, etc.)
   - Year
   - Transmission
   - Mileage
   - Fuel Type
   - Road Tax
   - MPG
   - Engine Size
3. Click "Predict Price"
4. View the predicted price and insights

### Exploring Data

- **Data Explorer Tab**: View dataset, statistics, and visualizations
- **Model Performance Tab**: Compare model metrics and performance

## 📊 Dataset

The dataset contains Audi car listings with the following features:
- **model**: Car model (A1, A3, A4, Q3, Q5, etc.)
- **year**: Year of manufacture
- **price**: Price in GBP (target variable)
- **transmission**: Manual or Automatic
- **mileage**: Mileage in miles
- **fuelType**: Petrol, Diesel, or Hybrid
- **tax**: Annual road tax
- **mpg**: Miles per gallon
- **engineSize**: Engine size in liters

## 🔧 Technical Details

### Preprocessing Steps
1. **Label Encoding**: Convert model and fuel type to numerical values
2. **One-Hot Encoding**: Convert transmission to binary columns
3. **Feature Scaling**: Standardize features using StandardScaler
4. **Train-Test Split**: 80-20 split with random_state=0

### Models Trained
- **Random Forest**: Ensemble of decision trees
- **Linear Regression**: Simple linear model
- **Extra Trees**: Randomized decision trees

### Evaluation Metrics
- **R² Score**: Coefficient of determination (higher is better)
- **MAE**: Mean Absolute Error in GBP (lower is better)

```

### Module Not Found Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## 📈 Model Performance

Typical results:
- **R² Score**: 0.92-0.96 (excellent fit)
- **MAE**: £800-£1200 (good accuracy)

