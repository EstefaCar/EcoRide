import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(layout="wide")

try:
    modelo_churn = joblib.load('modelo-churn.pkl')
    pipeline_preproc = joblib.load('pipeline-preproc.pkl')
except FileNotFoundError:
    st.error("Error: archivos del modelo no encontrados. Directorio actual: " + os.getcwd())
    st.stop()

st.title("EcoRide · Predicción de Churn")
st.write("Completa los datos del cliente para estimar la probabilidad de baja.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.slider("Edad (años)", 18, 80, 35)
    dias_antiguedad = st.slider("Antigüedad (días)", 0, 1825, 365)

with col2:
    uso_mensual_km = st.slider("Uso mensual (km)", 0.0, 1000.0, 150.0)
    gasto_promedio = st.slider("Gasto promedio mensual (€)", 0.0, 500.0, 50.0)

with col3:
    soporte_tickets = st.slider("Tickets de soporte (mes)", 0, 20, 1)
    plan = st.selectbox("Plan", ["basico", "elite", "premium"])

region = st.selectbox("Región", ["Centro", "Norte", "Sur"])

st.markdown("---")

if st.button("Predecir Churn"):
    input_data = pd.DataFrame({
        'Edad':              [edad],
        'Uso_Mensual_Km':    [uso_mensual_km],
        'Soporte_Tickets':   [soporte_tickets],
        'Gasto_Promedio':    [gasto_promedio],
        'Dias_Antiguedad':   [dias_antiguedad],
        'Plan':              [plan],
        'Region':            [region],
    })

    try:
        processed_input = pipeline_preproc.transform(input_data)
        churn_prediction = modelo_churn.predict(processed_input)
        churn_probability = modelo_churn.predict_proba(processed_input)[:, 1]

        prob = churn_probability[0]
        if churn_prediction[0] == 1:
            st.error(f"⚠️ **El cliente probablemente abandonará el servicio.**  \nProbabilidad de churn: **{prob:.0%}**")
        else:
            st.success(f"✅ **El cliente probablemente continuará usando el servicio.**  \nProbabilidad de churn: **{prob:.0%}**")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
