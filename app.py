import os
import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import pickle
import numpy as np
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]

# Set page config
st.set_page_config(
    page_title="Bank Subscription Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load models, scaler, and features
@st.cache_resource
def load_models():
    models_path = "models.pkl"
    features_path = "features.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(models_path):
        st.error(
            "Model file not found. Run train_model.py in this folder to generate models.pkl, scaler.pkl and features.pkl."
        )
        st.stop()
    
    if not os.path.exists(features_path):
        st.error(
            "Feature list not found. Run train_model.py in this folder to generate features.pkl."
        )
        st.stop()
    
    if not os.path.exists(scaler_path):
        st.error(
            "Scaler file not found. Run train_model.py in this folder to generate scaler.pkl."
        )
        st.stop()
    
    with open(models_path, "rb") as f:
        models = pickle.load(f)
    
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return models, features, scaler

models_dict, features, scaler = load_models()

# Header
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🏦 Bank Marketing Subscription Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 16px; color: #666;'>"  
    "Predict whether a customer is likely to subscribe to a term deposit using Machine Learning."  
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar with better organization
st.sidebar.markdown("<h2 style='color: #1f77b4;'>📋 Customer Profile</h2>", unsafe_allow_html=True)

age = st.sidebar.slider("Age", 18, 95, 35)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "other"
    ]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.sidebar.selectbox(
    "Education",
    ["primary", "secondary", "tertiary"]
)

balance = st.sidebar.number_input(
    "Account Balance",
    value=1000
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    ["yes", "no"]
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    ["yes", "no"]
)

day = st.sidebar.slider(
    "Last Contact Day",
    1,
    31,
    15
)

month = st.sidebar.selectbox(
    "Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ]
)

campaign = st.sidebar.slider(
    "Campaign Contacts",
    1,
    50,
    2
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    value=0
)

previous = st.sidebar.number_input(
    "Previous Contacts",
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "success",
        "failure",
        "other"
    ]
)

# Create input dataframe
input_dict = {
    "age": age,
    "balance": balance,
    "day": day,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
}

input_df = pd.DataFrame([input_dict])

# One-Hot Encode
input_df[f"job_{job}"] = 1
input_df[f"marital_{marital}"] = 1
input_df[f"education_{education}"] = 1
input_df[f"housing_{housing}"] = 1
input_df[f"loan_{loan}"] = 1
input_df[f"month_{month}"] = 1
input_df[f"poutcome_{poutcome}"] = 1

# Match training features
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features]

# Main content area
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Info", "👤 Customer Summary"])

# TAB 1: Prediction
with tab1:
    st.markdown("<h3 style='color: #1f77b4;'>Make a Prediction</h3>", unsafe_allow_html=True)
    
    col_model, col_predict = st.columns([2, 1])
    
    with col_model:
        selected_model = st.radio(
            "Select Model:",
            options=["random_forest", "logistic_regression"],
            format_func=lambda x: "🌳 Random Forest" if x == "random_forest" else "📈 Logistic Regression"
        )
    
    with col_predict:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔍 Predict", use_container_width=True, key="predict_btn")
    
    if predict_button:
        model = models_dict[selected_model]["model"]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create prediction result box
        if prediction == 1:
            st.markdown(
                f"<div class='prediction-box success-box'>"
                f"<h3 style='color: #28a745; margin: 0;'>✅ Likely to Subscribe</h3>"
                f"<p style='color: #155724; font-size: 18px; margin: 10px 0 0 0;'>"
                f"Confidence: <b>{probability:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='prediction-box danger-box'>"
                f"<h3 style='color: #dc3545; margin: 0;'>❌ Unlikely to Subscribe</h3>"
                f"<p style='color: #721c24; font-size: 18px; margin: 10px 0 0 0;'>"
                f"Confidence: <b>{probability:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Probability gauge chart
        st.markdown("<h4>Prediction Confidence</h4>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Subscription Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 33], 'color': "#fee5d9"},
                    {'range': [33, 66], 'color': "#fcae91"},
                    {'range': [66, 100], 'color': "#fb6a4a"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        )])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        with st.expander("📈 Detailed Analysis"):
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.metric("Model Used", "🌳 Random Forest" if selected_model == "random_forest" else "📈 Logistic Regression")
                st.metric("Model Accuracy", f"{models_dict[selected_model]['accuracy']:.2%}")
            
            with insight_col2:
                st.metric("Features Used", len(features))
                st.metric("Prediction", "Yes" if prediction == 1 else "No")

# TAB 2: Model Information
with tab2:
    st.markdown("<h3 style='color: #1f77b4;'>Model Information</h3>", unsafe_allow_html=True)
    
    col_rf, col_lr = st.columns(2)
    
    with col_rf:
        st.markdown("### 🌳 Random Forest")
        st.info(f"**Test Accuracy:** {models_dict['random_forest']['accuracy']:.2%}")
        st.markdown("""
        - Ensemble learning method
        - Multiple decision trees
        - Good at capturing non-linear patterns
        - Robust to outliers
        """)
    
    with col_lr:
        st.markdown("### 📈 Logistic Regression")
        st.info(f"**Test Accuracy:** {models_dict['logistic_regression']['accuracy']:.2%}")
        st.markdown("""
        - Linear classification model
        - Fast and interpretable
        - Works well with linear relationships
        - Efficient for large datasets
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    col_feat, col_samples = st.columns(2)
    
    with col_feat:
        st.metric("Total Features", len(features))
    
    with col_samples:
        st.metric("Feature Names", "Bank Customer Data")
    
    with st.expander("View Feature List"):
        st.write(features)

# TAB 3: Customer Summary
with tab3:
    st.markdown("<h3 style='color: #1f77b4;'>Customer Profile Summary</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{age} years")
    
    with col2:
        st.metric("Account Balance", f"€{balance:,}")
    
    with col3:
        st.metric("Campaign Contacts", campaign)
    
    with col4:
        st.metric("Previous Contacts", previous)
    
    st.markdown("---")
    st.markdown("### 📋 Full Input Data")
    
    # Create a nice display of the input
    summary_data = {
        "Age": [age],
        "Job": [job],
        "Marital Status": [marital],
        "Education": [education],
        "Account Balance": [f"€{balance:,}"],
        "Housing Loan": [housing.upper()],
        "Personal Loan": [loan.upper()],
        "Last Contact Day": [day],
        "Contact Month": [month.upper()],
        "Campaign Contacts": [campaign],
        "Days Since Previous": [pdays],
        "Previous Contacts": [previous],
        "Previous Outcome": [poutcome]
    }
    
    summary_df = pd.DataFrame(summary_data).T
    summary_df.columns = ["Value"]
    
    st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 12px;'>"
    "Built with Streamlit • Powered by Scikit-Learn • ML Models for Bank Telemarketing"
    "</p>",
    unsafe_allow_html=True
)
