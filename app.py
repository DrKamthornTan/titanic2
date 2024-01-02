import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define function to make prediction
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Load saved model and scaler
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Create input data array
    input_data = np.array([pclass, sex, age, sibsp, parch, fare, embarked])
    input_data = input_data.reshape(1, -1)

    # One-hot encode categorical features
    sex_encoder = OneHotEncoder(categories=[['female', 'male']])
    embarked_encoder = OneHotEncoder(categories=[['C', 'Q', 'S']])
    sex_embarked_encoded = np.concatenate([sex_encoder.fit_transform(input_data[:, [1]]).toarray(),
                                             embarked_encoder.fit_transform(input_data[:, [6]]).toarray()], axis=1)

    # Scale numerical features
    num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    input_data_numeric = input_data[:, [0, 2, 3, 4, 5]]
    input_data_numeric = scaler.transform(input_data_numeric)

    # Combine one-hot encoded and scaled numerical features
    input_data_scaled = np.concatenate([input_data_numeric, sex_embarked_encoded], axis=1)

    # Make prediction using model
    prediction = model.predict(input_data_scaled)[0]

    return prediction

# Set up page layout
st.set_page_config(page_title="Titanic Survival Prediction",
                   page_icon=":ship:",
                   layout="wide")

# Add banner image
image = Image.open("ship.jpg")
st.image(image, use_column_width=True)

# Add header
st.write("# Titanic Survival Prediction")

# Add form for user input
with st.form(key="input_form"):
    # Add input fields for passenger data
    st.write("## Passenger Information")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Add button to submit form
    submit_button = st.form_submit_button(label="Predict Survival")

# Make prediction when form is submitted
if submit_button:
    prediction = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == 1:
        st.write("### Prediction: **Survived**")
    else:
        st.write("### Prediction: **Did not survive**")