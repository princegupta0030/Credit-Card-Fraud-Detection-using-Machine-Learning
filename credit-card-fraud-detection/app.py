import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & HEADER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.markdown("""
Welcome to the Credit Card Fraud Detection portal.
Enter the transaction details below to evaluate the likelihood of fraud.

**Disclaimer:** *This tool provides a Machine Learning prediction based on historical data.
It is not a definitive determination of fraud and should be used alongside human review.*
""")

# -----------------------------------------------------------------------------
# 2. LOAD MODELS & CONFIGURATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """
    Load the trained model, scaler, and configuration exactly once per session
    to avoid retraining or reloading from disk on every interaction.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')

    # Load configuration
    with open(os.path.join(models_dir, 'final_config.json'), 'r') as f:
        config = json.load(f)

    # Load model and scaler
    model = joblib.load(os.path.join(models_dir, config['model_file']))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

    return model, scaler, config

try:
    model, scaler, config = load_assets()
except Exception as e:
    st.error(f"Failed to load required model assets. Please ensure the training pipeline has been run. Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. UI FOR USER INPUT
# -----------------------------------------------------------------------------
st.sidebar.header("Model Information")
st.sidebar.info(f"**Model:** {config['model_name']}")
st.sidebar.info(f"**Classification Threshold:** {config['selected_threshold']}")
st.sidebar.markdown(f"*{config['justification']}*")

st.header("Transaction Details")

# Layout for features: Amount is standard, V1-V28 are PCA transformed
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Standard Features")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)

    # Optional: If feature engineering was used during training (Log_Amount or Hour),
    # we would collect them here. Our base baseline drops Time, so we don't need Time.
    # Note: Our `app.py` ensures we pass EXACTLY the columns the model expects.
    # From our training, X_train has ['V1', 'V2', ..., 'V28', 'Amount']

with col2:
    st.subheader("Anonymized PCA Features (V1 - V28)")
    with st.expander("Expand to enter V1-V28 values"):
        pca_inputs = {}
        # Creating a neat grid for 28 inputs
        cols = st.columns(4)
        for i in range(1, 29):
            feature_name = f"V{i}"
            with cols[(i - 1) % 4]:
                pca_inputs[feature_name] = st.number_input(f"{feature_name}", value=0.0, format="%.4f")

# -----------------------------------------------------------------------------
# 4. PREDICTION LOGIC
# -----------------------------------------------------------------------------
if st.button("Evaluate Transaction", type="primary"):
    with st.spinner("Analyzing transaction..."):
        try:
            # 1. Construct DataFrame in the exact order used during training
            # The original raw data has V1-V28, then Amount, then Class
            input_dict = {f"V{i}": pca_inputs[f"V{i}"] for i in range(1, 29)}
            input_dict["Amount"] = amount

            df_input = pd.DataFrame([input_dict])

            # 2. Apply EXACT preprocessing
            # Our preprocessor expects 'Amount' to be scaled.
            # We scale only the Amount column using the fitted scaler
            df_input_scaled = df_input.copy()

            # Since the scaler was fitted on a subset of columns, we must ensure
            # we pass the exact structure it expects.
            # In our data_preprocessing.py, scaler was applied directly to a dataframe containing ONLY 'Amount'
            df_input_scaled[['Amount']] = scaler.transform(df_input[['Amount']])

            # 3. Generate Predictions
            proba_fraud = model.predict_proba(df_input_scaled)[0, 1]
            threshold = config['selected_threshold']

            is_fraud = proba_fraud >= threshold

            # 4. Display Results
            st.markdown("---")
            st.header("Evaluation Result")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if is_fraud:
                    st.error("🚨 **Potential Fraud**")
                else:
                    st.success("✅ **Likely Legitimate**")

            with res_col2:
                st.metric(label="Calculated Probability of Fraud", value=f"{proba_fraud * 100:.2f}%")
                st.caption(f"Decision threshold set at {threshold * 100:.2f}%")

            if is_fraud:
                st.warning("This transaction exceeds the risk threshold and requires further manual review or verification.")
            else:
                st.info("This transaction falls within normal operational parameters.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
