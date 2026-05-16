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
    # Input for Antiguedad_meses, will be converted to Dias_Antiguedad
    antiguedad_meses_input = st.slider("Antigüedad (meses)", 0, 60, 12)
    # New input for Edad
    edad = st.slider("Edad (años)", 18, 90, 30)

with col2:
    viajes_promedio_mes = st.slider("Viajes promedio por mes", 0.0, 50.0, 10.0)
    # Input for Kilometros_promedio_mes, will be mapped to Uso_Mensual_Km
    kilometros_promedio_mes = st.slider("Kilómetros promedio por mes", 0.0, 500.0, 50.0)
    # Input for Gasto_promedio_mes, will be mapped to Gasto_Promedio
    gasto_promedio_mes = st.slider("Gasto promedio por mes (€)", 0.0, 200.0, 30.0)

with col3:
    # Input for Reclamaciones_mes, will be mapped to Soporte_Tickets
    reclamaciones_mes = st.slider("Reclamaciones por mes", 0, 5, 0)
    sub_plan_options = ['Básico', 'Estándar', 'Premium']
    # Input for Sub_plan, will be mapped to Plan
    sub_plan = st.selectbox("Plan de suscripción", sub_plan_options)
    # New input for Region
    region_options = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro'] # Placeholder options, adjust as needed
    region = st.selectbox("Región", region_options)


metodo_pago_options = ['Tarjeta de crédito', 'PayPal', 'Transferencia bancaria']
metodo_pago = st.selectbox("Método de pago", metodo_pago_options)

tipo_vehiculo_options = ['Patinete', 'Bicicleta', 'Moto']
tipo_vehiculo = st.selectbox("Tipo de vehículo", tipo_vehiculo_options)

tiene_seguro = st.checkbox("Tiene seguro")

st.markdown("---")

if st.button("Predecir Churn"):
    # Create a DataFrame from inputs, ensuring column names match the pipeline's expectations
    input_data = pd.DataFrame({
        'Dias_Antiguedad': [antiguedad_meses_input * 30], # Convert months to days
        'Edad': [edad],
        'Gasto_Promedio': [gasto_promedio_mes],
        'Region': [region],
        'Uso_Mensual_Km': [kilometros_promedio_mes],
        'Soporte_Tickets': [reclamaciones_mes],
        'Plan': [sub_plan],
        # Keeping these as they were in the original app, assuming they are handled or dropped by the pipeline
        'viajes_promedio_mes': [viajes_promedio_mes],
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
