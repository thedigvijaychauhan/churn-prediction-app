import streamlit as st
import pickle
import pandas as pd

# load model
model = pickle.load(open("model/model.pkl", "rb"))

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.markdown("### Enter Customer Details")

# INPUTS
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly Charges", 0.0, 10000.0)
    total = st.number_input("Total Charges", 0.0, 100000.0)

# PREDICTION
if st.button("🚀 Predict Churn"):

    # create empty df with exact columns
    input_df = pd.DataFrame(columns=model.feature_names_in_)
    input_df.loc[0] = 0

    # numeric values
    if "tenure" in input_df.columns:
        input_df["tenure"] = tenure

    if "MonthlyCharges" in input_df.columns:
        input_df["MonthlyCharges"] = monthly

    if "TotalCharges" in input_df.columns:
        input_df["TotalCharges"] = total

    # categorical safe mapping
    mappings = [
        f"gender_{gender}",
        f"Partner_{partner}",
        f"Dependents_{dependents}",
        f"PhoneService_{phone}",
        f"InternetService_{internet}",
        f"Contract_{contract}"
    ]

    for col in mappings:
        if col in input_df.columns:
            input_df[col] = 1

    # prediction
    prediction = model.predict(input_df)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("❌ Customer will churn")
    else:
        st.success("✅ Customer will stay")

    st.balloons()