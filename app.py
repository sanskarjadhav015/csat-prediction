import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==========================================
# 1. APP CONFIGURATION & TITLE
# ==========================================
st.set_page_config(page_title="CSAT Predictor", layout="centered")
st.title("🛒 eCommerce Customer Satisfaction (CSAT) Predictor")

st.markdown("""
This application predicts the **Customer Satisfaction Score (CSAT)** for eCommerce support interactions. 
It uses interaction parameters such as communication channel, issue category, agent shift, and response time.
""")

# ==========================================
# 2. SIDEBAR: USER INPUT FEATURES
# ==========================================
st.sidebar.header("Input Interaction Details")

def get_user_input():
    # Categorical Inputs based on notebook EDA
    channel_name = st.sidebar.selectbox("Communication Channel", ["Inbound", "Outcall"])
    category = st.sidebar.selectbox("Issue Category", ["Product Queries", "Order Related", "Returns", "Cancellation", "Refund Related", "Feedback", "Other"])
    product_category = st.sidebar.selectbox("Product Category", ["Electronics", "Apparel", "Home Appliances", "Beauty", "Unknown"])
    agent_shift = st.sidebar.selectbox("Agent Shift", ["Morning", "Afternoon", "Evening", "Night", "Split"])
    tenure_bucket = st.sidebar.selectbox("Agent Tenure", ["0-30", "31-60", "61-90", ">90", "On Job Training"])
    
    # Time Inputs to calculate 'response_time' (Feature engineered in notebook)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Timestamps")
    reported_time = st.sidebar.text_input("Issue Reported Time (YYYY-MM-DD HH:MM)", "2023-08-01 10:00")
    responded_time = st.sidebar.text_input("Issue Responded Time (YYYY-MM-DD HH:MM)", "2023-08-01 10:45")
    
    # Calculate response time in minutes
    try:
        fmt = "%Y-%m-%d %H:%M"
        t1 = datetime.strptime(reported_time, fmt)
        t2 = datetime.strptime(responded_time, fmt)
        response_time_mins = (t2 - t1).total_seconds() / 60.0
    except ValueError:
        response_time_mins = 0.0 # Default fallback if format is incorrect
        st.sidebar.warning("Please enter dates in YYYY-MM-DD HH:MM format.")

    # Store inputs in a dictionary
    user_data = {
        'channel_name': channel_name,
        'category': category,
        'Product_category': product_category,
        'Agent Shift': agent_shift,
        'Tenure Bucket': tenure_bucket,
        'response_time': response_time_mins
    }
    
    return pd.DataFrame(user_data, index=[0])

# Fetch user inputs
input_df = get_user_input()

# Display User Inputs
st.subheader("Selected Interaction Features")
st.write(input_df)

# ==========================================
# 3. MODEL PREDICTION LOGIC
# ==========================================
st.markdown("---")
st.subheader("Predict CSAT Score")

# Note to user on how to connect the notebook model
st.info("""
**Developer Note:** To make live predictions, ensure you export your trained model (e.g., Decision Tree), PCA transformer, and One-Hot Encoders from your `.ipynb` notebook using `joblib` or `pickle`, and load them here.
""")

if st.button("Predict CSAT"):
    try:
        # ---------------------------------------------------------
        # UNCOMMENT AND UPDATE THIS SECTION ONCE MODELS ARE SAVED
        # ---------------------------------------------------------
        
        # 1. Load the pre-trained artifacts
        # model = joblib.load('best_csat_model.pkl')
        # encoder = joblib.load('one_hot_encoder.pkl')
        # pca = joblib.load('pca_transformer.pkl')
        
        # 2. Preprocess the input_df (Apply encoding, handling 'Unknown' values)
        # encoded_features = encoder.transform(input_df[['channel_name', 'category', 'Product_category', 'Agent Shift', 'Tenure Bucket']])
        # response_time_arr = np.array(input_df['response_time']).reshape(-1, 1)
        # final_features = np.hstack((encoded_features, response_time_arr))
        
        # 3. Apply PCA Dimensionality Reduction (n_components=10 as per notebook)
        # final_features_pca = pca.transform(final_features)
        
        # 4. Predict
        # prediction = model.predict(final_features_pca)
        # st.success(f"🎯 The predicted Customer Satisfaction (CSAT) Score is: **{prediction[0]} / 5**")
        
        # --- Dummy Output for UI testing ---
        st.success("🎯 The predicted Customer Satisfaction (CSAT) Score is: **4 / 5** (Mock Prediction)")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")