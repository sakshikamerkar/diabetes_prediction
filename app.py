import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("Diabetes Prediction Using Patient Data")
st.write("Machine Learning Mini Project")

# Load saved models
saved_data = joblib.load("diabetes_models.pkl")
models = saved_data["models"]
features = saved_data["features"]

st.subheader("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model",
    ["Random Forest", "Neural Network"]
)

st.subheader("Enter Patient Measurements")

glucose = st.number_input(
    "Glucose Level", min_value=0.0, value=120.0
)

bp = st.number_input(
    "Blood Pressure", min_value=0.0, value=70.0
)

skin = st.number_input(
    "Skin Thickness", min_value=0.0, value=20.0
)

insulin = st.number_input(
    "Insulin Level", min_value=0.0, value=80.0
)

bmi = st.number_input(
    "BMI", min_value=0.0, value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age", min_value=1, value=30
)

if st.button("Predict"):
    sample = pd.DataFrame([[
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]], columns=features)

    # Select the trained model
    model = models[model_name]

    # Predict class and estimated probability
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1] * 100

    st.subheader("Prediction Result")
    st.write("Selected model:", model_name)

    st.metric(
        "Model-estimated probability",
        f"{probability:.2f}%"
    )

    if prediction == 1:
        st.warning("Predicted class: Diabetic")
    else:
        st.success("Predicted class: Non-diabetic")

    st.caption(
        "Educational project only. This output is not "
        "a medical diagnosis or a validated assessment "
        "of an individual's diabetes risk."
    )