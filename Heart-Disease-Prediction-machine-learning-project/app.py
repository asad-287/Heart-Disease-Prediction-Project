import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prediction function (same as your Flask app)
def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose):
    male_encoded = 1 if male.lower() == 'male' else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == 'yes' else 0
    BPMeds_encoded = 1 if BPMeds.lower() == 'yes' else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == 'yes' else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == 'yes' else 0
    diabetes_encoded = 1 if diabetes.lower() == 'yes' else 0

    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded,
                          prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)
    return "The Patient has Heart Disease" if result[0] == 1 else "The Patient has No Heart Disease"

# Streamlit app
st.title("Heart Disease Prediction Form")

# Form for user input
with st.form(key='prediction_form'):
    col1, col2 = st.columns(2)

    with col1:
        male = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        currentSmoker = st.selectbox("Current Smoker", ["Yes", "No"])
        cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0.0, step=1.0)
        BPMeds = st.selectbox("BP Medications", ["Yes", "No"])
        prevalentStroke = st.selectbox("Prevalent Stroke", ["Yes", "No"])
        prevalentHyp = st.selectbox("Prevalent Hypertension", ["Yes", "No"])

    with col2:
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        totChol = st.number_input("Total Cholesterol", min_value=0.0, step=1.0)
        sysBP = st.number_input("Systolic BP", min_value=0.0, step=1.0)
        diaBP = st.number_input("Diastolic BP", min_value=0.0, step=1.0)
        BMI = st.number_input("BMI", min_value=0.0, step=0.01)
        heartRate = st.number_input("Heart Rate", min_value=0.0, step=1.0)
        glucose = st.number_input("Glucose Level", min_value=0.0, step=1.0)

    submit_button = st.form_submit_button(label="Predict")

# Display prediction result
if submit_button:
    try:
        prediction = predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                            diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
