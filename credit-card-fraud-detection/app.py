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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')

    with open(os.path.join(models_dir, 'final_config.json'), 'r') as f:
        config = json.load(f)

    model = joblib.load(os.path.join(models_dir, config['model_file']))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

    return model, scaler, config

try:
    model, scaler, config = load_assets()
except Exception as e:
    st.error(f"Failed to load required model assets. Please ensure the training pipeline has been run. Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. AUTO-FILL SIMULATOR (To avoid manual typing)
# -----------------------------------------------------------------------------
if 'amount' not in st.session_state:
    st.session_state.amount = 100.0
for i in range(1, 29):
    if f'V{i}' not in st.session_state:
        st.session_state[f'V{i}'] = 0.0

def fill_fraud():
    st.session_state.amount = 0.0
    st.session_state.V1 = -2.3122
    st.session_state.V2 = 1.9519
    st.session_state.V3 = -1.6098
    st.session_state.V4 = 3.9979
    st.session_state.V5 = -0.5221
    st.session_state.V6 = -1.4265
    st.session_state.V7 = -2.5373
    st.session_state.V8 = 1.3916
    st.session_state.V9 = -2.7700
    st.session_state.V10 = -2.7722
    st.session_state.V11 = 3.2020
    st.session_state.V12 = -2.8999
    st.session_state.V13 = -0.5952
    st.session_state.V14 = -4.2892
    st.session_state.V15 = 0.3897
    st.session_state.V16 = -1.1407
    st.session_state.V17 = -2.8300
    for i in range(18, 29):
         st.session_state[f'V{i}'] = 0.0

def fill_normal():
    st.session_state.amount = 150.0
    st.session_state.V1 = 1.2
    st.session_state.V2 = 0.1
    for i in range(3, 29):
         st.session_state[f'V{i}'] = 0.0

st.header("🧪 Quick Test Simulator")
st.write("Typing 28 abstract numbers is annoying! Use these buttons to instantly auto-fill real examples from the dataset:")

col_a, col_b = st.columns(2)
with col_a:
    st.button("🟢 Auto-Fill Legitimate Transaction", on_click=fill_normal, use_container_width=True)
with col_b:
    st.button("🔴 Auto-Fill Fraudulent Transaction", on_click=fill_fraud, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. UI FOR USER INPUT
# -----------------------------------------------------------------------------
st.sidebar.header("Model Information")
st.sidebar.info(f"**Model:** {config['model_name']}")
st.sidebar.info(f"**Classification Threshold:** {config['selected_threshold']}")
st.sidebar.markdown(f"*{config['justification']}*")

st.header("Transaction Details")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Standard Features")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=10.0, key='amount')

with col2:
    st.subheader("Anonymized PCA Features (V1 - V28)")
    with st.expander("Expand to view/edit V1-V28 values"):
        pca_inputs = {}
        cols = st.columns(4)
        for i in range(1, 29):
            feature_name = f"V{i}"
            with cols[(i - 1) % 4]:
                pca_inputs[feature_name] = st.number_input(f"{feature_name}", format="%.4f", key=f"V{i}")

# -----------------------------------------------------------------------------
# 5. PREDICTION LOGIC
# -----------------------------------------------------------------------------
if st.button("Evaluate Transaction", type="primary", use_container_width=True):
    with st.spinner("Analyzing transaction..."):
        try:
            input_dict = {f"V{i}": pca_inputs[f"V{i}"] for i in range(1, 29)}
            input_dict["Amount"] = amount

            df_input = pd.DataFrame([input_dict])
            df_input_scaled = df_input.copy()
            df_input_scaled[['Amount']] = scaler.transform(df_input[['Amount']])

            proba_fraud = model.predict_proba(df_input_scaled)[0, 1]
            threshold = config['selected_threshold']

            is_fraud = proba_fraud >= threshold

            st.markdown("---")
            st.header("Evaluation Result")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if is_fraud:
                    st.error("🚨 **Potential Fraud Detected**")
                else:
                    st.success("✅ **Transaction Likely Legitimate**")

            with res_col2:
                st.metric(label="Calculated Probability of Fraud", value=f"{proba_fraud * 100:.2f}%")
                st.caption(f"Decision threshold set at {threshold * 100:.2f}%")

            if is_fraud:
                st.warning("This transaction exceeds the risk threshold and requires further manual review or verification.")
            else:
                st.info("This transaction falls within normal operational parameters.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
