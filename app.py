import streamlit as st
import pandas as pd
import joblib

# --- UI/UX Design & Color Theory ---
# Using a clean, modern aesthetic with a focus on data analysis colors and clear warnings.
# Streamlit's default theme is clean, but custom CSS will enhance the visual cues.

st.set_page_config(
    page_title="EcoRide: Predicción de Churn",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a dark, professional theme with green accents and red warnings
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #1a1a1a; /* Dark gray/nearly black */
        color: #e0e0e0; /* Light gray for general text */
    }
    /* Headers */
    .main-header {color: #28a745; text-align: center; font-size: 2.5em; margin-bottom: 20px;} /* EcoRide Green */
    .subheader {color: #e0e0e0; text-align: center; font-size: 1.5em; margin-bottom: 30px;}
    h1, h2, h3, h4, h5, h6 {color: #e0e0e0;}
    /* Streamlit components specific styling */
    .stButton>button {
        background-color: #007bff; /* Blue for action */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 1.1em;
        cursor: pointer;
    }
    .stButton>button:hover {background-color: #0056b3; color: white;}
    .stTextInput>div>div>input, .stSelectbox>div>div>div>div, .stNumberInput>div>div>input, .stTextArea>div>div>textarea, .stRadio > label {
        background-color: #333333; /* Darker input background */
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #555555; /* Subtle border */
        padding: 8px;
    }
    /* Checkbox styling */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] {color: #e0e0e0;}
    /* Specific styling for success/error messages */
    .stAlert.success {
        background-color: #28a745; /* Green for success */
        color: white;
        border-radius: 8px;
    }
    .stAlert.error {
        background-color: #dc3545; /* Red for error/churn */
        color: white;
        border-radius: 8px;
    }
    .stAlert.info {
        background-color: #007bff; /* Blue for info */
        color: white;
        border-radius: 8px;
    }
    /* Sidebar styling */
    .css-1d391kg { /* This might be the sidebar content wrapper, depends on streamlit version */
        background-color: #2a2a2a; /* Slightly lighter dark for sidebar */
        color: #e0e0e0;
    }
    /* Adjust padding for content */
    .main .block-container {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    /* Markdown text color in general */
    .stMarkdown p {color: #e0e0e0;}
    </style>
    """,
    unsafe_allow_html=True
)


# --- Load Models ---
# The models are expected to be in the current working directory, which is /content/drive/MyDrive/Ejercicios IA/EcoRide
try:
    with open('modelo-churn.pkl', 'rb') as f:
        modelo_churn = joblib.load(f)
    with open('pipeline-preproc.pkl', 'rb') as f:
        pipeline_preproc = joblib.load(f)
    st.sidebar.success("Modelos de predicción cargados exitosamente!")
except Exception as e:
    st.sidebar.error(f"Error cargando modelos: {e}. Asegúrate de que 'modelo-churn.pkl' y 'pipeline-preproc.pkl' estén en la carpeta correcta: /content/drive/MyDrive/Ejercicios IA/EcoRide")
    st.stop() # Stop the app if models cannot be loaded


# --- App Title and Description ---
st.markdown('<h1 class="main-header">EcoRide 🚲</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Dashboard de Predicción de Churn</h2>', unsafe_allow_html=True)

st.write(
    """
Bienvenido al panel de predicción de churn de EcoRide. Esta herramienta utiliza modelos de Machine Learning para identificar clientes con alto riesgo de cancelar su suscripción. Ingresa los datos del cliente a continuación para obtener una predicción.
"""
)

# --- Input Features ---
st.subheader("Datos del Cliente")

# Numerical Inputs
col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad del Cliente", min_value=18, max_value=90, value=30, help="Edad del cliente en años.")
    uso_mensual_km = st.number_input("Uso Mensual (Km)", min_value=0.0, max_value=5000.0, value=150.0, format="%.1f", help="Kilómetros recorridos mensualmente por el cliente.")
    soporte_tickets = st.number_input("Tickets de Soporte (últimos 3 meses)", min_value=0, max_value=30, value=1, help="Número de tickets de soporte generados por el cliente en los últimos 3 meses.")
with col2:
    gasto_promedio = st.number_input("Gasto Promedio Mensual ($)", min_value=0.0, max_value=500.0, value=75.0, format="%.2f", help="Gasto promedio mensual del cliente en el servicio.")
    dias_antiguedad = st.number_input("Días de Antigüedad", min_value=0, max_value=3650, value=365, help="Número de días que el cliente lleva con el servicio.")
    plan = st.radio("Plan Actual", ['Básico', 'Estándar', 'Premium'], help="Tipo de plan de suscripción del cliente.")

# Region Input (will be one-hot encoded)
region = st.radio("Región del Cliente", ['Centro', 'Norte', 'Sur'], help="Región geográfica donde reside el cliente.")


# --- Prediction Button ---
st.markdown("\n---") # Separator
if st.button("Predecir Churn", key="predict_button"):

    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'Edad': [edad],
        'Uso_Mensual_Km': [uso_mensual_km],
        'Soporte_Tickets': [soporte_tickets],
        'Gasto_Promedio': [gasto_promedio],
        'Dias_Antiguedad': [dias_antiguedad],
        'Plan': [plan],
        'Region': [region] # Pass a single 'Region' column, let the pipeline handle one-hot encoding
    })

    try:
        # Preprocess the input data using the loaded pipeline
        # IMPORTANT: Ensure the pipeline_preproc can handle the raw 'Plan' and one-hot encoded regions correctly.
        # If the pipeline expects 'Plan' to be one-hot encoded or label encoded, you might need to adjust this.
        processed_data = pipeline_preproc.transform(input_data)

        # Make prediction
        prediction = modelo_churn.predict(processed_data)
        prediction_proba = modelo_churn.predict_proba(processed_data)[:, 1] # Probability of churning

        st.subheader("\nResultado de la Predicción:")
        if prediction[0] == 1: # Assuming 1 means churn
            st.error(f"¡Alerta! Este cliente tiene un ALTO riesgo de **CANCELAR** su suscripción con una probabilidad del {prediction_proba[0]*100:.2f}%.")
            st.markdown("<p style='color: #e0e0e0;'>**Recomendación:** Considera contactar al cliente proactivamente con ofertas personalizadas o soporte para retenerlo.</p>", unsafe_allow_html=True)
        else:
            st.success(f"Este cliente es probable que **NO CANCELE** su suscripción, con una probabilidad del {(1 - prediction_proba[0])*100:.2f}%.")
            st.markdown("<p style='color: #e0e0e0;'>**Recomendación:** Sigue monitoreando su satisfacción y fomenta su lealtad.</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al realizar la predicción. Por favor, verifica los datos de entrada y la compatibilidad con el pipeline: {e}")

# --- Footer and Sidebar Information ---
st.sidebar.title("Acerca de EcoRide")
st.sidebar.info(
    "EcoRide es una empresa de micromovilidad urbana comprometida con soluciones de transporte sostenibles y eficientes. "
    "Este dashboard es una herramienta interna para mejorar la retención de clientes mediante la predicción de churn."
)
st.sidebar.image("https://i.imgur.com/G55n90H.png", width=200) # Placeholder for a more fitting logo
st.sidebar.markdown("--- \n _Powered by Google Colab_ ")
