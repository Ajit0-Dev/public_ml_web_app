# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 00:10:42 2025

@author: admin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open("trained_model.sav", "rb"))
heart_disease_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        default_index=0
    )

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Level")
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Level")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age of the Person")

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        try:
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diab_prediction = diabetes_model.predict([input_data])

            if diab_prediction[0] == 1:
                diab_diagnosis = "The person is diabetic"
            else:
                diab_diagnosis = "The person is not diabetic"
        except ValueError:
            diab_diagnosis = "⚠️ Please enter all fields as numeric values."

    st.success(diab_diagnosis)

# Heart Disease Prediction
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting ECG results')
    with col2:
        thalach = st.text_input('Max Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST Depression induced by Exercise')
    with col2:
        slope = st.text_input('Slope of Peak Exercise ST Segment')
    with col3:
        ca = st.text_input('Major Vessels Colored by Fluoroscopy')
    with col1:
        thal = st.text_input('Thal: 0=normal; 1=fixed defect; 2=reversible defect')

    heart_diagnosis = ""

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            heart_prediction = heart_disease_model.predict([input_data])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'
        except ValueError:
            heart_diagnosis = "⚠️ Please enter all fields as numeric values."

    st.success(heart_diagnosis)

# Parkinsons Prediction
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    inputs = []
    for i in range(0, len(features), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(features):
                val = cols[j].text_input(features[i + j])
                inputs.append(val)

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            input_data = [float(i) for i in inputs]
            parkinsons_prediction = parkinsons_model.predict([input_data])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = "⚠️ Please enter all fields as numeric values."

    st.success(parkinsons_diagnosis)
