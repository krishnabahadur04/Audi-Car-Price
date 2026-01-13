"""
Car Price Prediction - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Audi Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF6B6B;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all saved models and preprocessing objects"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder_model.pkl', 'rb') as f:
            le_model = pickle.load(f)
        with open('label_encoder_fuel.pkl', 'rb') as f:
            le_fuel = pickle.load(f)
        with open('column_transformer.pkl', 'rb') as f:
            ct = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, le_model, le_fuel, ct, scaler, metadata
    except FileNotFoundError:
        st.error("Model files not found! Please run train_model.py first.")
        st.stop()

# Load data
@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('audi.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure audi.csv is in the same directory.")
        st.stop()

# Load everything
model, le_model, le_fuel, ct, scaler, metadata = load_models()
df = load_data()

# Header
st.title("🚗 Audi Car Price Prediction")
st.markdown("### Predict the price of an Audi car based on its features")

# Sidebar
with st.sidebar:
    st.image("https://www.carlogos.org/car-logos/audi-logo-2016.png", width=150)
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info(f"**Model:** {metadata['best_model']}")
    st.success(f"**R² Score:** {metadata['r2_score']:.4f}")
    st.warning(f"**MAE:** £{metadata['mae']:.2f}")
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Statistics")
    st.write(f"Total Records: {len(df):,}")
    st.write(f"Price Range: £{df['price'].min():,.0f} - £{df['price'].max():,.0f}")
    st.write(f"Average Price: £{df['price'].mean():,.0f}")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "📈 Model Performance"])

# Tab 1: Prediction
with tab1:
    st.markdown("### Enter Car Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get unique models from the dataset
        model_options = sorted(df['model'].unique())
        car_model = st.selectbox("🚙 Model", model_options)
        
        year = st.slider("📅 Year", 
                        min_value=int(df['year'].min()), 
                        max_value=int(df['year'].max()), 
                        value=int(df['year'].median()))
        
        mileage = st.number_input("🛣️ Mileage (miles)", 
                                  min_value=0, 
                                  max_value=200000, 
                                  value=15000, 
                                  step=1000)
    
    with col2:
        transmission = st.selectbox("⚙️ Transmission", 
                                   sorted(df['transmission'].unique()))
        
        fuel_type = st.selectbox("⛽ Fuel Type", 
                                sorted(df['fuelType'].unique()))
        
        tax = st.number_input("💰 Road Tax (£)", 
                             min_value=0, 
                             max_value=600, 
                             value=150, 
                             step=10)
    
    with col3:
        mpg = st.number_input("⛽ MPG (Miles Per Gallon)", 
                             min_value=10.0, 
                             max_value=100.0, 
                             value=50.0, 
                             step=0.1)
        
        engine_size = st.selectbox("🔧 Engine Size (L)", 
                                   sorted(df['engineSize'].unique()))
    
    st.markdown("---")
    
    if st.button("🔮 Predict Price", type="primary"):
        # Prepare input data
        input_data = np.array([[car_model, year, transmission, mileage, 
                               fuel_type, tax, mpg, engine_size]], dtype=object)
        
        # Apply preprocessing
        try:
            # Label encode model and fuel type
            input_processed = input_data.copy()
            input_processed[0, 0] = le_model.transform([car_model])[0]
            input_processed[0, 4] = le_fuel.transform([fuel_type])[0]
            
            # Create feature array for transformation
            X_input = input_processed[:, [0, 1, 2, 3, 4, 5, 6, 7]]

            # Apply column transformer (one-hot encoding)
            X_transformed = ct.transform(X_input)
            
            # Apply scaling
            X_scaled = scaler.transform(X_transformed)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Price</h2>
                    <h1>£{prediction:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                similar_cars = df[
                    (df['model'] == car_model) & 
                    (df['year'] == year)
                ]
                avg_similar = similar_cars['price'].mean() if len(similar_cars) > 0 else df['price'].mean()
                st.metric("Average Similar Cars", f"£{avg_similar:,.2f}")
            
            with col2:
                diff_pct = ((prediction - avg_similar) / avg_similar * 100) if avg_similar > 0 else 0
                st.metric("Difference from Average", f"{diff_pct:+.1f}%")
            
            with col3:
                percentile = (df['price'] <= prediction).sum() / len(df) * 100
                st.metric("Price Percentile", f"{percentile:.1f}%")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure all inputs are valid.")

# Tab 2: Data Explorer
with tab2:
    st.markdown("### Dataset Overview")
    
    # Display data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(100), use_container_width=True)
    
    with col2:
        st.markdown("#### Quick Stats")
        st.write(f"**Total Cars:** {len(df):,}")
        st.write(f"**Models:** {df['model'].nunique()}")
        st.write(f"**Years:** {df['year'].min()} - {df['year'].max()}")
        st.write(f"**Avg Price:** £{df['price'].mean():,.0f}")
        st.write(f"**Avg Mileage:** {df['mileage'].mean():,.0f}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price Distribution")
        fig = px.histogram(df, x='price', nbins=50, 
                          title='Distribution of Car Prices',
                          labels={'price': 'Price (£)'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Price by Fuel Type")
        fig = px.box(df, x='fuelType', y='price', 
                    title='Price Distribution by Fuel Type',
                    labels={'price': 'Price (£)', 'fuelType': 'Fuel Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Average Price by Model")
        avg_by_model = df.groupby('model')['price'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(avg_by_model, 
                    title='Top 10 Models by Average Price',
                    labels={'value': 'Average Price (£)', 'model': 'Model'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Price vs Year")
        fig = px.scatter(df.sample(min(1000, len(df))), x='year', y='price', 
                        color='fuelType',
                        title='Price vs Year (Sample)',
                        labels={'price': 'Price (£)', 'year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Performance
with tab3:
    st.markdown("### Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Best Model</h3>
                <h2>{}</h2>
            </div>
        """.format(metadata['best_model']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>R² Score</h3>
                <h2>{:.4f}</h2>
            </div>
        """.format(metadata['r2_score']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>Mean Absolute Error</h3>
                <h2>£{:.2f}</h2>
            </div>
        """.format(metadata['mae']), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### All Models Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in metadata['all_results'].items():
        comparison_data.append({
            'Model': model_name,
            'R² Score': metrics['r2'],
            'MAE (£)': metrics['mae']
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('R² Score', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = px.bar(comparison_df, x='Model', y='R² Score', 
                    title='Model R² Score Comparison',
                    color='R² Score',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Model Interpretation")
    
    st.info("""
    **R² Score (Coefficient of Determination):** Indicates how well the model explains the variance in car prices. 
    A score closer to 1.0 means better predictions.
    
    **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual prices. 
    Lower values indicate more accurate predictions.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🚗 Audi Car Price Prediction System | Built with Streamlit</p>
        <p>Data-driven predictions using machine learning</p>
    </div>
""", unsafe_allow_html=True)
