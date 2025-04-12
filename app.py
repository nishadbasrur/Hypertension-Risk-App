import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Hypertension Risk Estimator", layout="centered")

st.title("🩺 Hypertension Risk Estimator")
st.write("This app estimates your likelihood of having hypertension based on basic health indicators. For educational use only.")

# Load dataset and train model in memory
@st.cache_data
def load_and_train():
    df = pd.read_csv("framingham.csv")
    df.dropna(inplace=True)
    
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_and_train()

# User inputs
age = st.slider('Age', 20, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Do you currently smoke?", ["Yes", "No"])
cigs_per_day = st.number_input("Cigarettes per day", min_value=0, max_value=60, value=0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
sysBP = st.number_input("Systolic Blood Pressure", min_value=90.0, max_value=200.0, value=120.0)
diaBP = st.number_input("Diastolic Blood Pressure", min_value=60.0, max_value=140.0, value=80.0)
totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=400.0, value=200.0)
heart_rate = st.number_input("Heart Rate", min_value=40.0, max_value=140.0, value=70.0)
bpm_meds = st.selectbox("Are you taking BP medication?", ["Yes", "No"])
diabetes = st.selectbox("Do you have diabetes?", ["Yes", "No"])

# Create feature vector
features = [
    1 if sex == "Male" else 0,
    age,
    1 if smoker == "Yes" else 0,
    cigs_per_day,
    1 if bpm_meds == "Yes" else 0,
    1 if diabetes == "Yes" else 0,
    totChol,
    sysBP,
    diaBP,
    bmi,
    heart_rate,
    sysBP - diaBP,  # Pulse Pressure
    (1 if smoker == "Yes" else 0) * cigs_per_day  # Smoke Intensity
]

# Estimate risk
if st.button("Estimate My Risk"):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    st.markdown(f"**Estimated Risk Probability:** {probability:.1%}")

    if prediction == 1:
        st.error("⚠️ You are likely to have hypertension.")
        st.markdown("""
        #### Next Steps:
        - Schedule a check-up with your primary care physician.
        - Monitor your blood pressure at home regularly.
        - Reduce sodium in your diet and avoid processed foods.
        - Get 30 minutes of physical activity most days.
        - Resources:
          - [CDC High Blood Pressure Guide](https://www.cdc.gov/bloodpressure/)
          - [American Heart Association](https://www.heart.org/en/health-topics/high-blood-pressure)
        """)
    else:
        st.success("✅ You are likely NOT to have hypertension.")
        st.markdown("""
        #### Keep It Up:
        - Maintain a healthy lifestyle with balanced meals and regular exercise.
        - Avoid smoking and limit alcohol intake.
        - Check blood pressure annually as a precaution.
        - Resources:
          - [Healthy Living Tips - Mayo Clinic](https://www.mayoclinic.org/healthy-lifestyle)
          - [NIH Heart, Lung, and Blood Institute](https://www.nhlbi.nih.gov/health-topics/high-blood-pressure)
        """)

# Footer disclaimer
st.markdown(
    """
    <hr style='margin-top: 40px; margin-bottom:10px;'>
    <div style='font-size: 0.8em; color: gray; text-align: center'>
        ⚠️ <b>Disclaimer:</b> This tool is intended for informational and educational purposes only.<br>
        It does not diagnose, treat, or provide medical advice.<br>
        Always consult a licensed healthcare provider before making health decisions.<br><br>
        Built by [Nishad B], 2025 • Based on the Framingham Heart Study Dataset
    </div>
    """,
    unsafe_allow_html=True
)
