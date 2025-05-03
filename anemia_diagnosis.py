import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="anemia_diagnosis.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Cek Risiko Anemia")
st.write("Masukkan data sederhana untuk mengetahui apakah seseorang berisiko mengalami anemia atau tidak.")

# Form input pengguna
sex = st.selectbox("Jenis Kelamin (Laki-laki / Perempuan)", options=["Male", "Female"])
red = st.number_input("Persentase Warna Merah pada Citra Darah (%)", min_value=0.0, max_value=100.0, value=35.0)
green = st.number_input("Persentase Warna Hijau pada Citra Darah (%)", min_value=0.0, max_value=100.0, value=30.0)
blue = st.number_input("Persentase Warna Biru pada Citra Darah (%)", min_value=0.0, max_value=100.0, value=35.0)
hb = st.number_input("Kadar Hemoglobin (Hb) dalam darah (g/dL)", min_value=0.0, max_value=20.0, value=12.0)

if st.button("Cek Hasil"):
    # Preprocessing
    sex_encoded = 1 if sex == "Male" else 0
    input_data = np.array([[sex_encoded, red, green, blue, hb]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Model inference
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Ambil hasil prediksi
    predicted_label = np.argmax(prediction)
    result = label_encoder.inverse_transform([predicted_label])[0]

    # Tampilkan hasil
    if result.lower() == "anemia":
        st.error("Hasil: Berisiko Mengalami Anemia")
    else:
        st.success("Hasil: Tidak Berisiko Mengalami Anemia")
