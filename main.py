import streamlit as st
import joblib
import numpy as np


st.title("Deteksi Penyakit Diabetes")
st.caption("Aplikasi Deteksi Penyakit Diabetes")


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
    
    
    
    
    st.text(f"Hasil Analisa: {'Berpotensi Diabetes.' if prediksi[0] == 1 else 'Tidak Berpotensi Diabetes.'}")
    
