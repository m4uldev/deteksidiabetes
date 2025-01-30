import streamlit as st
import joblib
import numpy as np
import google.generativeai as genai

st.title("Deteksi Penyakit Diabetes")
st.caption("Aplikasi Deteksi Penyakit Diabetes")

# Konfigurasi API Gemini
genai.configure(api_key=st.secrets["auth_token"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_ai = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Input pengguna
kehamilan = st.number_input('Kehamilan')
glukosa = st.number_input('Glukosa')
tekanan_darah = st.number_input("Tekanan Darah")
ketebalan_kulit = st.number_input("Ketebalan Kulit")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
fungsi_riwayat_diabetes = st.number_input("Fungsi Riwayat Diabetes")
usia = st.number_input("Usia")

# Tombol Prediksi
kalkulasi_diabetes = st.button("Cek Diabetes")
if kalkulasi_diabetes:
    model = joblib.load('./model/model_deteksi_diabetes.joblib')
    data_input = np.array([[kehamilan, glukosa, tekanan_darah, ketebalan_kulit, insulin, bmi, fungsi_riwayat_diabetes, usia]])
    prediksi = model.predict(data_input)
    
    if prediksi[0] == 1:
        hasil = "Berpotensi Diabetes."
        prompt = "Berikan tips untuk mengelola dan mengobati diabetes secara alami dan medis."
    else:
        hasil = "Tidak Berpotensi Diabetes."
        prompt = "Bagaimana cara menjaga kesehatan agar terhindar dari diabetes?"
    
    response = model_ai.generate_content(prompt)
    
    st.text(f"Hasil Analisa: {hasil}")
    st.text("Tips:")
    st.write(response.text)
