
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config for better layout
st.set_page_config(layout="wide")

# Load the churn model and preprocessing pipeline for deployment
# These files are expected to be in the same directory as app.py on streamlit.io
try:
    # Adjust path if models are in a subdirectory, e.g., 'models/modelo-churn.pkl'
    modelo_churn = joblib.load('modelo-churn.pkl')
    pipeline_preproc = joblib.load('pipeline-preproc.pkl')
except FileNotFoundError:
    st.error("Error: Model or pipeline files not found. Make sure 'modelo-churn.pkl' and 'pipeline-preproc.pkl' are in the same directory as app.py. Current working directory: " + os.getcwd())
    st.stop()

st.title("EcoRide Churn Prediction App")
st.write("Enter customer details to predict churn likelihood.")

st.markdown("--- Jardín de entrada para la predicción de bajas de clientes --- ")

# Create columns for better layout of input fields
col1, col2, col3 = st.columns(3)

with col1:
    antiguedad_meses = st.slider("Antigüedad (meses)", 0, 60, 12)
    viajes_promedio_mes = st.slider("Viajes promedio por mes", 0.0, 50.0, 10.0)

with col2:
    kilometros_promedio_mes = st.slider("Kilómetros promedio por mes", 0.0, 500.0, 50.0)
    gasto_promedio_mes = st.slider("Gasto promedio por mes (€)", 0.0, 200.0, 30.0)

with col3:
    reclamaciones_mes = st.slider("Reclamaciones por mes", 0, 5, 0)
    sub_plan_options = ['Básico', 'Estándar', 'Premium']
    sub_plan = st.selectbox("Plan de suscripción", sub_plan_options)

metodo_pago_options = ['Tarjeta de crédito', 'PayPal', 'Transferencia bancaria']
metodo_pago = st.selectbox("Método de pago", metodo_pago_options)

tipo_vehiculo_options = ['Patinete', 'Bicicleta', 'Moto']
tipo_vehiculo = st.selectbox("Tipo de vehículo", tipo_vehiculo_options)

tiene_seguro = st.checkbox("Tiene seguro")

st.markdown("---")

if st.button("Predecir Churn"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'antiguedad_meses': [antiguedad_meses],
        'viajes_promedio_mes': [viajes_promedio_mes],
        'kilometros_promedio_mes': [kilometros_promedio_mes],
        'gasto_promedio_mes': [gasto_promedio_mes],
        'reclamaciones_mes': [reclamaciones_mes],
        'sub_plan': [sub_plan],
        'metodo_pago': [metodo_pago],
        'tipo_vehiculo': [tipo_vehiculo],
        'tiene_seguro': [int(tiene_seguro)] # Convert boolean to int (0 or 1)
    })

    # Use the loaded pipeline and model
    try:
        processed_input = pipeline_preproc.transform(input_data)
        churn_prediction = modelo_churn.predict(processed_input)
        churn_probability = modelo_churn.predict_proba(processed_input)[:, 1] # Probability of churn (class 1)

        if churn_prediction[0] == 1:
            st.error(f"**Predicción:** El cliente probablemente abandonará el servicio. **Probabilidad:** {churn_probability[0]:.2f}")
        else:
            st.success(f"**Predicción:** El cliente probablemente NO abandonará el servicio. **Probabilidad:** {churn_probability[0]:.2f}")
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.write("Por favor, verifique que el modelo y el pipeline se hayan cargado correctamente y que los datos de entrada coincidan con sus expectativas.")

