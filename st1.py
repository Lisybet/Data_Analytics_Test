import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# =========================
# Configuración inicial
# =========================
load_dotenv(".env.gemini")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

st.title("📊 Análisis inteligente de datos")

# Subida de archivo CSV
uploaded_file = st.file_uploader("📎 Envie su archivo CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    st.subheader("Vista previa de los datos")
    st.dataframe(df)

    # =========================
    # Pregunta libre sobre los datos
    # =========================
    st.subheader("🤖 Pregunta libre sobre los datos")
    pregunta_usuario = st.text_input("Haz tu pregunta sobre los datos")

    if st.button("Responder"):
        if pregunta_usuario:
            # Convertir DataFrame a JSON
            datos_json = df.to_dict(orient="records")  # lista de diccionarios
            datos_json_str = json.dumps(datos_json, indent=2)

            prompt = f"""
Eres un analista experto en datos.

Aquí están los datos completos en formato JSON:
{datos_json_str}

Pregunta del usuario: {pregunta_usuario}
"""
            respuesta = llm.invoke(prompt)
            st.markdown(respuesta.content)