# Membangun Model Machine Learning

Repository ini berisi implementasi proses pelatihan dan evaluasi model machine learning sebagai bagian dari tugas submission Machine Learning.

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model klasifikasi menggunakan algoritma **Random Forest** untuk memprediksi kelas obesitas berdasarkan data kesehatan dan gaya hidup. Proses pengembangan model mencakup tahapan pemisahan data, hyperparameter tuning, evaluasi performa, serta pencatatan eksperimen menggunakan **MLflow**.

## Dataset
Dataset yang digunakan merupakan dataset yang telah melalui tahap preprocessing dan disimpan dalam file:


Target klasifikasi pada dataset ini adalah kolom **`NObeyesdad`**, sedangkan fitur lainnya digunakan sebagai variabel input model.

## Metode yang Digunakan
- Algoritma: **Random Forest Classifier**
- Pembagian data: **80% data latih dan 20% data uji**
- Hyperparameter tuning: **GridSearchCV**
- Metode evaluasi:
  - Accuracy
  - F1-score (weighted)
  - Confusion Matrix

## Eksperimen dan Tracking
Eksperimen model dicatat menggunakan **MLflow**, termasuk:
- Parameter terbaik hasil tuning
- Nilai metrik evaluasi (accuracy dan F1-score)
- Model terbaik
- Artifact pendukung seperti confusion matrix dan feature importance

Untuk kebutuhan eksperimen, MLflow dapat terintegrasi dengan **DagsHub** sebagai remote tracking. Pada pipeline CI, eksperimen dijalankan secara non-interaktif untuk memastikan proses pelatihan model berjalan otomatis dan reproducible.

## Otomatisasi dengan GitHub Actions
Repository ini menggunakan **GitHub Actions** untuk menjalankan proses training dan evaluasi model secara otomatis setiap kali terjadi perubahan pada branch utama. Pipeline ini memastikan bahwa:
- Proses pelatihan model dapat dijalankan tanpa interaksi manual
- Dependensi terinstal dengan benar
- Script pelatihan dapat dieksekusi dengan konsisten

## Cara Menjalankan Program
Jalankan perintah berikut untuk melatih dan mengevaluasi model:

```bash
python modelling_tuning.py
