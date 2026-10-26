import streamlit as st
import numpy as np
import joblib

# Load the saved model and preprocessing tools
try:
    model = joblib.load('f1_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_driver = joblib.load('le_driver.pkl')
    le_team = joblib.load('le_team.pkl')
except Exception as e:
    st.error("Error loading model files. Please ensure the .pkl files are in the repository.")

st.title("🏎️ Formula 1 Podium Predictor")
st.write("Predict whether a driver will finish in the Top 3 based on starting conditions.")

# UI Input Elements
col1, col2 = st.columns(2)

with col1:
    # Get the list of driver and team names directly from the encoders
    driver = st.selectbox("Select Driver", le_driver.classes_)
    team = st.selectbox("Select Constructor (Team)", le_team.classes_)

with col2:
    circuit_id = st.number_input("Circuit ID (e.g., 1 for Albert Park)", min_value=1, max_value=80, value=1)
    grid_pos = st.number_input("Starting Grid Position", min_value=1, max_value=24, value=1)

if st.button("Predict Podium Finish"):
    # 1. Encode the text inputs to numbers
    driver_encoded = le_driver.transform([driver])[0]
    team_encoded = le_team.transform([team])[0]
    
    # 2. Format the features exactly as trained: ['circuitId', 'grid', 'driver_encoded', 'constructor_encoded']
    input_features = np.array([[circuit_id, grid_pos, driver_encoded, team_encoded]])
    
    # 3. Scale the features
    input_scaled = scaler.transform(input_features)
    
    # 4. Make Prediction
    prediction = model.predict(input_scaled)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success(f"🏆 YES! The model predicts {driver} will finish on the podium.")
    else:
        st.error(f"❌ NO. The model predicts {driver} will finish outside the top 3.")
