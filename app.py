import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("final1_model.pkl", "rb"))
scaler = pickle.load(open("scaler1.pkl", "rb"))

st.title("🎯Customer Churn Prediction")

# Input fields

age = st.number_input("Age", min_value=18, max_value=100)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
gender = st.selectbox("Gender", ["Female", "Male"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Map input to numeric
gender_val = 1 if gender == "Male" else 0

# One-hot encoding for contract
contract_month = 1 if contract == "Month-to-month" else 0
contract_one = 1 if contract == "One year" else 0
# contract_two is drop_first

# One-hot encoding for payment method
payment_elec = 1 if payment_method == "Electronic check" else 0
payment_mail = 1 if payment_method == "Mailed check" else 0
payment_bank = 1 if payment_method == "Bank transfer" else 0
# payment_credit is drop_first

# Combine features
input_features = np.array([[age, tenure, monthly_charges, total_charges,
                            gender_val,
                            contract_month, contract_one,
                            payment_elec, payment_mail, payment_bank]])

# Scale
input_scaled = scaler.transform(input_features)

# Predict
if st.button("Predict Churn"):

    # prediction
    prediction = model.predict(input_scaled)[0]

    # probabilities
    probs = model.predict_proba(input_scaled)[0]

    churn_prob = probs[1]
    no_churn_prob = probs[0]

    # Display Result
    st.success(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")

    st.subheader("📊 Prediction Confidence")

    st.write(f"No Churn: {no_churn_prob*100:.2f}%")
    st.write(f"Churn: {churn_prob*100:.2f}%")

    # Bar chart
    chart_data = {
        "No Churn": no_churn_prob,
        "Churn": churn_prob
    }

    st.bar_chart(chart_data)
