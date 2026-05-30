import streamlit as st
import requests
import os

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Use environment variable (best practice for Docker)
API_URL = os.getenv("API_URL", "http://backend:5000")

st.title("🧠 Multiple Disease Prediction System")

st.sidebar.title("Navigation")

choice = st.sidebar.radio(
    "Select Disease",
    ["Diabetes", "Heart Disease", "Kidney Disease"]
)

# ---------------- SAFE REQUEST FUNCTION ----------------
def get_prediction(endpoint, payload):
    try:
        res = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        return res
    except Exception as e:
        return None, str(e)

# ---------------- DIABETES ----------------
if choice == "Diabetes":
    st.header("Diabetes Prediction")

    with st.form("diabetes_form"):
        gender = st.selectbox("Gender", [0, 1])
        age = st.number_input("Age", 1, 120, 40)
        hp = st.selectbox("Hypertension", [0, 1])
        hd = st.selectbox("Heart Disease", [0, 1])
        smoke = st.selectbox("Smoking", [0, 1])
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
        hba1c = st.number_input("HbA1c", 3.0, 20.0, 5.5)
        glucose = st.number_input("Glucose", 50, 400, 100)

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "features": [gender, age, hp, hd, smoke, bmi, hba1c, glucose]
        }

        res = requests.post(f"{API_URL}/predict/diabetes", json=payload)

        if res.status_code == 200:
            try:
                st.success("Prediction Successful")
                st.json(res.json())
            except:
                st.error("Invalid JSON response")
                st.text(res.text)
        else:
            st.error(f"Request failed: {res.status_code}")
            st.text(res.text)

# ---------------- HEART ----------------
elif choice == "Heart Disease":
    st.header("Heart Disease Prediction")

    with st.form("heart_form"):
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        bp = st.number_input("Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "features": [age, sex, cp, bp, chol]
        }

        res = requests.post(f"{API_URL}/predict/heart", json=payload)

        if res.status_code == 200:
            try:
                st.success("Prediction Successful")
                st.json(res.json())
            except:
                st.error("Invalid JSON response")
                st.text(res.text)
        else:
            st.error(f"Request failed: {res.status_code}")
            st.text(res.text)

# ---------------- KIDNEY ----------------
else:
    st.header("Kidney Disease Prediction")

    with st.form("kidney_form"):
        age = st.number_input("Age", 1, 120, 50)
        bp = st.number_input("Blood Pressure", 50, 200, 80)
        sc = st.number_input("Creatinine", 0.1, 20.0, 1.0)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 12.0)

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "features": [age, bp, sc, hemo]
        }

        res = requests.post(f"{API_URL}/predict/kidney", json=payload)

        if res.status_code == 200:
            try:
                st.success("Prediction Successful")
                st.json(res.json())
            except:
                st.error("Invalid JSON response")
                st.text(res.text)
        else:
            st.error(f"Request failed: {res.status_code}")
            st.text(res.text)