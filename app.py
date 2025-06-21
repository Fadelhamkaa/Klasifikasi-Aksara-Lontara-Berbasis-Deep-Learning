import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd  
# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Klasifikasi Aksara Lontara",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Aplikasi Klasifikasi Aksara Lontara")
st.write(
    "Upload sebuah gambar aksara Lontara (ka, ga, nga, dll.) "
    "dan model akan memprediksi aksara apakah itu."
)

# --- FUNGSI UNTUK MEMUAT MODEL (DENGAN CACHING) ---
# @st.cache_resource agar model hanya di-load sekali
@st.cache_resource
def load_my_model():
    """Memuat model H5 yang sudah dilatih."""
    model = tf.keras.models.load_model('lontara_classifier_model.h5')
    return model

# Panggil fungsi untuk memuat model
model = load_my_model()

# Definisikan nama kelas sesuai urutan folder saat training
# Pastikan urutan ini sama dengan 'train_generator.class_indices'
class_names = ['a', 'ba', 'ca', 'da', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'mpa', 'na', 'nca', 'nga', 'ngka', 'nra', 'nya', 'pa', 
               'ra', 'sa', 'ta', 'wa', 'ya']

# --- FUNGSI UNTUK PREPROCESSING GAMBAR ---
def preprocess_image(image_file):
    """Mengubah file gambar menjadi format yang sesuai untuk model."""
    # Buka gambar menggunakan Pillow
    img = Image.open(image_file)
    
    # Ubah ke mode RGB jika gambar memiliki alpha channel (PNG)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize gambar sesuai input model
    img = img.resize((128, 128))
    
    # Ubah gambar menjadi numpy array dan normalisasi
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


# --- INTERFACE UPLOAD GAMBAR ---
uploaded_file = st.file_uploader(
    "Pilih gambar aksara...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Tampilkan gambar yang diupload di kolom kiri
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diupload', use_container_width=True)

    # Lakukan prediksi di kolom kanan
    with col2:
        st.write("Menganalisis gambar...")
        
        # Preprocess gambar
        processed_image = preprocess_image(uploaded_file)
        
        # Lakukan prediksi
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence_score = np.max(prediction) * 100
        
        st.success(f"**Prediksi:** {predicted_class_name}")
        st.info(f"**Tingkat Keyakinan:** {confidence_score:.2f}%")
        st.write("---")
        st.write("Probabilitas untuk setiap kelas:")
        
        # <--- 2. BLOK KODE INI MENGGANTIKAN 'st.bar_chart(prediction[0])'
        
        # Buat label custom sesuai format yang Anda inginkan: 'nama(indeks)'
        custom_labels = [f"{name}({i})" for i, name in enumerate(class_names)]

        # Buat DataFrame Pandas dari hasil prediksi
        df_probs = pd.DataFrame({
            'Probabilitas': prediction[0]
        }, index=custom_labels)

        # Tampilkan DataFrame sebagai bar chart
        st.bar_chart(df_probs)