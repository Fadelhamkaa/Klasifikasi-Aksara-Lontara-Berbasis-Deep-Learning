# Klasifikasi Aksara Lontara Berbasis Deep Learning

### Aplikasi Web untuk Mengenali Aksara Bugis Kuno Menggunakan TensorFlow & Streamlit

![Status Proyek](https://img.shields.io/badge/status-selesai-green)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)

Aplikasi ini adalah implementasi dari model *Convolutional Neural Network* (CNN) yang dilatih untuk mengklasifikasikan 23 aksara dasar dalam tulisan Lontara. Proyek ini bertujuan untuk menunjukkan penerapan praktis *computer vision* dalam melestarikan warisan budaya digital, dari tahap pemodelan hingga deployment aplikasi web yang fungsional.

---

### ▶️ Demo Aplikasi

**[[LIHAT LIVE DEMO DI SINI]](https://[LINK-KE-APLIKASI-STREAMLIT-ANDA].streamlit.app/)** *(Ganti link ini dengan URL aplikasi Anda yang sudah di-deploy)*

---

### 🖼️ Tampilan Aplikasi

*(! PENTING: Ganti gambar di bawah dengan screenshot terbaru aplikasi Anda yang **menampilkan grafik dengan label custom**)*

![Screenshot Aplikasi Lontara](https://i.imgur.com/example.png) 
*(Contoh Screenshot)*

---

## 📜 Tentang Proyek
Aksara Lontara adalah sistem tulisan tradisional yang digunakan oleh suku Bugis-Makassar di Sulawesi Selatan. Seiring modernisasi, penggunaan aksara ini semakin berkurang. Proyek ini dibangun sebagai inisiatif kecil untuk menjembatani teknologi modern dengan warisan budaya.

Tujuan utama dari proyek ini adalah:
1.  Membangun dan melatih model *Deep Learning* yang akurat untuk klasifikasi gambar aksara Lontara.
2.  Mendeploy model tersebut ke dalam sebuah aplikasi web interaktif yang mudah diakses dan digunakan oleh siapa saja.
3.  Menjadi portofolio *end-to-end* yang menunjukkan kemampuan dalam pengumpulan data, *preprocessing*, pemodelan, hingga *deployment*.

---

## ✨ Fitur Utama
* **Klasifikasi Gambar Interaktif**: Pengguna dapat mengupload gambar aksara Lontara (JPG, PNG, JPEG) untuk mendapatkan prediksi secara langsung.
* **Antarmuka Pengguna yang Bersih**: Dibangun dengan Streamlit untuk pengalaman pengguna yang sederhana dan intuitif di berbagai perangkat.
* **Visualisasi Probabilitas Detail**: Menampilkan grafik batang interaktif dengan label yang jelas (misal: 'ka(7)', 'pa(17)'), menunjukkan tingkat kepercayaan model untuk setiap kemungkinan aksara.
* **Performa Model Tinggi**: Model dilatih menggunakan teknik *Transfer Learning* untuk mencapai akurasi yang solid, bahkan dengan dataset yang terbatas.

---

## 🛠️ Tumpukan Teknologi (Tech Stack)
* **Bahasa Pemrograman**: Python
* **Machine Learning & Deep Learning**: TensorFlow, Keras
* **Analisis & Manipulasi Data**: NumPy, Pandas
* **Framework Aplikasi Web**: Streamlit
* **Pemrosesan Gambar**: Pillow (PIL)
* **Deployment**: Streamlit Community Cloud
* **Manajemen Kode**: Git & GitHub

---

## 📂 Struktur Proyek
Klasifikasi-Aksara-Lontara-Berbasis-Deep-Learning/
├── 📂 Dataset/           # Folder berisi data gambar (tidak di-commit ke Git)
│   ├── train/
│   └── val/
├── 📜 app.py             # Script utama aplikasi Streamlit
├── 📜 lontara_classifier_model.h5 # Model terlatih
├── 📜 README.md          # Dokumentasi proyek (file ini)
└── 📜 requirements.txt    # Daftar dependensi Python


---

## 🚀 Instalasi dan Cara Menjalankan Secara Lokal

Untuk menjalankan aplikasi ini di komputer lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/Fadelhamkaa/Klasifikasi-Aksara-Lontara-Berbasis-Deep-Learning.git](https://github.com/Fadelhamkaa/Klasifikasi-Aksara-Lontara-Berbasis-Deep-Learning.git)
    cd Klasifikasi-Aksara-Lontara-Berbasis-Deep-Learning
    ```

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Install semua dependensi yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

5.  Buka browser Anda dan akses alamat `http://localhost:8501`.

---

## 🧠 Arsitektur Model & Pelatihan
Model ini dibangun menggunakan pendekatan **Transfer Learning** dengan **MobileNetV2** sebagai model dasarnya.

* **Model Dasar**: MobileNetV2 yang telah dilatih pada dataset ImageNet. Lapisan konvolusinya dibekukan (*frozen*) untuk memanfaatkan fitur-fitur yang sudah dipelajarinya.
* **Lapisan Kustom**: Di atas model dasar, ditambahkan lapisan `GlobalAveragePooling2D`, `Dropout` (untuk regularisasi), dan sebuah lapisan `Dense` dengan 23 unit output dan aktivasi `softmax` untuk klasifikasi multikelas.
* **Dataset**: Dataset kustom yang terdiri dari 23 kelas aksara.
    * Data Latih: 80 gambar per kelas.
    * Data Validasi: 20 gambar per kelas.
* **Augmentasi Data**: Untuk mengatasi jumlah data yang terbatas dan mencegah *overfitting*, teknik augmentasi seperti rotasi, pergeseran, dan zoom diterapkan pada data latih.
* **Hasil**: Model berhasil mencapai akurasi validasi sekitar **[TULIS-AKURASI-VALIDASI-ANDA-DI-SINI, misal: 96%]%**. Model juga mampu memberikan prediksi dengan tingkat keyakinan tinggi pada data individual (contoh: 97.05% untuk aksara 'mpa').

*(! Opsional: Tambahkan gambar grafik akurasi & loss dari proses training Anda di sini)*
![Grafik Training](https://i.imgur.com/example_graph.png)

---

## 💡 Tantangan & Pembelajaran
* **Dataset Terbatas**: Mengatasi tantangan dataset yang kecil dengan efektif menggunakan *data augmentation* dan *transfer learning*, yang terbukti mampu menghasilkan model yang generalis.
* **Visualisasi Data yang Efektif**: Belajar memodifikasi output visualisasi standar (seperti `st.bar_chart`) menggunakan Pandas untuk menciptakan label yang lebih informatif dan *user-friendly* pada grafik.
* **Proses Deployment End-to-End**: Menguasai alur kerja lengkap dari menyimpan model Keras, membangun antarmuka dengan Streamlit, manajemen dependensi, hingga men-deploy-nya ke platform publik.

---

## 📞 Kontak
Jika Anda tertarik untuk berdiskusi lebih lanjut, jangan ragu untuk menghubungi saya.

* **Nama**: [Muhammad Fadel Hamka]
* **GitHub**: [https://github.com/Fadelhamkaa](https://github.com/Fadelhamkaa)
* **LinkedIn**: [https://linkedin.com/in/[PROFIL-LINKEDIN-ANDA]](https://www.linkedin.com/in/fadelhamka/]) *(Pastikan Anda menggunakan link profil LinkedIn yang benar di sini)*
