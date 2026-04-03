
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version warnings from unpickling the model
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Define the paths for the model and the data
MODEL_PATH = r'E:\projects\Bank note prediction\random_forest_model.pkl'
DATA_PATH = r'E:\projects\Bank note prediction\BankNote_Authentication.csv'

# --- Load the trained model --- 
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# --- Load data and fit scaler --- 
# This is necessary because the model was trained on scaled data.
# We need the same scaler to preprocess new input data.
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}")
    st.stop()

df_original = pd.read_csv(DATA_PATH)
X_for_scaler = df_original.drop('class', axis=1)

scaler = StandardScaler()
scaler.fit(X_for_scaler)

# --- Streamlit UI --- 
st.title('Bank Note Authentication Prediction')
st.write('Enter the values for the bank note features to get a prediction:')

# Input fields for features
variance = st.number_input('Variance', value=0.0)
skewness = st.number_input('Skewness', value=0.0)
curtosis = st.number_input('Curtosis', value=0.0)
entropy = st.number_input('Entropy', value=0.0)

# Prediction button
if st.button('Predict'):
    # Create a DataFrame from input values
    input_data = pd.DataFrame([{
        'variance': variance,
        'skewness': skewness,
        'curtosis': curtosis,
        'entropy': entropy
    }])

    # Scale the input data using the fitted scaler
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)

    # Display prediction
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.success(f"The bank note is likely Genuine. (Probability: {prediction_proba[0][0]:.2f})")
    else:
        st.error(f"The bank note is likely Forged. (Probability: {prediction_proba[0][1]:.2f})")

st.write("--- Developed with Streamlit and Random Forest Model ---")
