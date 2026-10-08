import streamlit as st
import numpy as np
import joblib

# ---------------------------
# Load saved model, scaler, and label encoder
# ---------------------------
# model = joblib.load("best_decision_tree.pkl")
# scaler = joblib.load("scaler.pkl")
# le = joblib.load("label_encoder.pkl")
model = joblib.load(r"C:\Users\abhin\Desktop\Abhinav\BMN_533\best_decision_tree.pkl")
scaler = joblib.load(r"C:\Users\abhin\Desktop\Abhinav\BMN_533\scaler.pkl")
le = joblib.load(r"C:\Users\abhin\Desktop\Abhinav\BMN_533\label_encoder.pkl")




st.title("Air Quality Index Prediction App")

# User inputs
year = st.number_input("Enter Year", min_value=2000, max_value=2030, value=2025)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=9)
pm25 = st.number_input("Enter PM2.5 value", min_value=0.0)
pm10 = st.number_input("Enter PM10 value", min_value=0.0)
no2 = st.number_input("Enter NO2 value", min_value=0.0)
so2 = st.number_input("Enter SO2 value", min_value=0.0)
o3 = st.number_input("Enter O3 value", min_value=0.0)

# Location input (binary: Varachha / Other)
location = st.selectbox("Select Location", ["Varachha", "Other"])
location_varachha = 1 if location == "Varachha" else 0

# Arrange features in same order as training
features = np.array([[year, month, pm25, pm10, no2, so2, o3, location_varachha]])

# Button for prediction
if st.button("Predict AQI Category"):
    # Debugging outputs
    st.write("Classes in model:", model.classes_)
    st.write("Classes in label encoder:", le.classes_)
    st.write("Feature before scaling:", features)
    st.write("Feature after scaling:", scaler.transform(features))

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict AQI Category
    prediction = model.predict(features_scaled)
    prediction_label = le.inverse_transform(prediction)[0]

    st.success(f"Predicted AQI Category: {prediction_label}")



# if __name__ == "__main__":
#     import os
#     os.system("streamlit run " + __file__)

