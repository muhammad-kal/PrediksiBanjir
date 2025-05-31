# Laporan Proyek Machine Learning - Haikal

## Domain Proyek

Banjir merupakan salah satu bencana alam yang sering terjadi di berbagai wilayah, termasuk Indonesia, dan dapat menyebabkan kerugian signifikan baik secara material maupun non-material. Keterlambatan dalam mendeteksi potensi banjir seringkali memperparah dampak yang ditimbulkan. Oleh karena itu, pengembangan sistem prediksi banjir yang akurat dan tepat waktu menjadi sangat penting. Dengan memanfaatkan data historis ketinggian air dari berbagai pos pemantauan, kita dapat membangun model machine learning yang mampu memberikan peringatan dini akan potensi terjadinya banjir. Proyek ini bertujuan untuk membuat model predictive analytics untuk memprediksi status banjir berdasarkan data-data tersebut.

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle: [Dataset Banjir Jakarta](https://www.kaggle.com/datasets/asfilianova/dataset-banjir).

## Business Understanding

### Problem Statements
1.  Bagaimana cara membangun model machine learning yang dapat memprediksi status potensi banjir (Banjir/Tidak Banjir) secara akurat berdasarkan data historis ketinggian air dari berbagai pos pemantauan?
2.  Algoritma machine learning manakah yang memberikan performa terbaik dalam melakukan klasifikasi status banjir pada dataset yang digunakan?

### Goals
1.  Mengembangkan model prediksi yang mampu mengklasifikasikan status banjir dengan tingkat akurasi, presisi, dan recall yang baik.
2.  Melakukan perbandingan performa antara beberapa algoritma machine learning (K-Nearest Neighbors, Decision Tree, Random Forest, SVM, dan Naive Bayes) untuk menemukan model yang paling optimal.

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang diajukan adalah sebagai berikut:
1.  Melakukan pra-pemrosesan data secara komprehensif yang mencakup pembersihan data, penanganan nilai yang hilang, dan transformasi fitur yang diperlukan.
2.  Menerapkan dan melatih beberapa algoritma klasifikasi machine learning (KNN, Decision Tree, Random Forest, SVM, dan Naive Bayes) pada data yang telah diproses.
3.  Mengevaluasi setiap model menggunakan metrik performa yang relevan (Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix) untuk memilih model terbaik sebagai solusi prediksi banjir.

## Data Understanding

Dataset yang digunakan adalah "pemetaan\_daerah\_banjir.csv". Dataset ini bersumber dari Kaggle ([Dataset Banjir Jakarta](https://www.kaggle.com/datasets/asfilianova/dataset-banjir)) dan berisi data historis pemantauan ketinggian air serta status banjir.
-   Jumlah sampel data awal: 624 baris
-   Jumlah fitur awal: 12 kolom

### Variabel-variabel pada dataset awal adalah sebagai berikut:
* `Tanggal`: Tanggal pencatatan data (tipe object). Merupakan tanggal spesifik saat data ketinggian air dan status banjir dicatat.
* `Waktu`: Waktu pencatatan data per jam (tipe object). Merupakan waktu spesifik pada tanggal pencatatan.
* `Katulampa`: Ketinggian air di pos pemantauan Katulampa dalam satuan cm (tipe int64). Pos ini sering menjadi indikator awal potensi banjir kiriman ke Jakarta.
* `Pos Depok`: Ketinggian air di pos pemantauan Depok dalam satuan cm (tipe int64). Merupakan pos setelah Katulampa di aliran sungai Ciliwung.
* `Manggarai`: Ketinggian air di pos pemantauan Pintu Air Manggarai dalam satuan cm (tipe float64). Pintu air ini krusial untuk pengendalian banjir di Jakarta.
* `Istiqlal`: Ketinggian air di pos pemantauan Istiqlal dalam satuan cm (tipe float64). Merepresentasikan ketinggian air di area pusat Jakarta.
* `Jembatan Merah`: Ketinggian air di pos pemantauan Jembatan Merah dalam satuan cm (tipe float64).
* `Flusing Ancol`: Ketinggian air di pos pemantauan Flusing Ancol dalam satuan cm (tipe float64). Area ini terkait dengan sistem drainase menuju laut.
* `Marina Ancol`: Ketinggian air di pos pemantauan Marina Ancol dalam satuan cm (tipe float64). Merepresentasikan kondisi di area pesisir.
* `Status Banjir`: Variabel target yang menunjukkan status banjir (tipe int64); 0 berarti Tidak Banjir, 1 berarti Banjir.
* `Unnamed: 10`: Kolom tanpa nama yang berisi data tambahan seperti keterangan sumber atau catatan level siaga (tipe object). 
* `Unnamed: 11`: Kolom tanpa nama yang berisi data tambahan serupa dengan `Unnamed: 10`, beberapa baris saja yang berisi level siaga banjir (tipe object), sisanya adalah kosong.

### Exploratory Data Analysis (EDA)
EDA dilakukan untuk mendapatkan pemahaman yang lebih baik tentang data:
1.  **Informasi Umum Data**: `data.info()` digunakan untuk melihat tipe data setiap kolom dan jumlah nilai non-null.
2.  **Statistik Deskriptif**: `data.describe()` digunakan untuk melihat statistik dasar fitur numerik.
3.  **Pengecekan Nilai Hilang**: `data.isnull().sum()` digunakan untuk mengidentifikasi jumlah nilai yang hilang pada setiap fitur. Ditemukan nilai hilang pada kolom 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', dan 'Marina Ancol', serta sebagian besar nilai hilang pada 'Unnamed: 10' dan 'Unnamed: 11'.
4.  **Distribusi Fitur Numerik**: Visualisasi histogram untuk setiap fitur numerik guna memahami distribusinya.
    5.  **Distribusi Variabel Target (`Status Banjir`)**: Visualisasi countplot menunjukkan distribusi kelas pada variabel target 'Status Banjir'.
    6.  **Heatmap Korelasi**: Untuk melihat hubungan linear antar fitur numerik setelah pembersihan awal.
    ## Data Preparation
Tahapan persiapan data yang dilakukan adalah sebagai berikut:
1.  **Pembersihan Kolom Tidak Relevan**: Kolom 'Unnamed: 10' dan 'Unnamed: 11' dihapus karena sebagian besar nilainya kosong dan kontennya (seperti keterangan sumber atau level siaga) tidak akan digunakan secara langsung dalam model prediksi numerik ini.
2.  **Penanganan Missing Values**: Nilai yang hilang pada kolom 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', dan 'Marina Ancol' diisi menggunakan nilai median dari masing-masing kolom. Median dipilih karena lebih robust terhadap nilai ekstrem (outlier) yang mungkin ada dalam data ketinggian air.
3.  **Penghapusan Fitur Waktu**: Kolom 'Tanggal' dan 'Waktu' dihapus dari dataset fitur. Meskipun informasi temporal ini berpotensi berguna, untuk model awal dan penyederhanaan, fitur ini dihilangkan agar fokus pada fitur numerik ketinggian air.
4.  **Pemisahan Fitur dan Target**: Dataset dibagi menjadi variabel fitur (X) yang berisi data ketinggian air dan variabel target (y) yang berisi 'Status Banjir'.
5.  **Pembagian Data (Train-Test Split)**: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari `sklearn.model_selection`. Penggunaan `random_state=42` memastikan reproduktifitas hasil pembagian. Parameter `stratify=y` juga digunakan untuk memastikan proporsi kelas target ('Status Banjir') sama pada data latih dan data uji, yang penting jika ada ketidakseimbangan kelas.
6.  **Penskalaan Fitur (Feature Scaling)**: Fitur-fitur numerik dalam data latih dan data uji diskalakan menggunakan `MinMaxScaler`. Penskalaan ini bertujuan untuk mengubah skala fitur ke rentang 0-1. Scaler di-'fit' *hanya* pada data latih (`X_train`) dan kemudian digunakan untuk mentransformasi baik data latih (`X_train`) maupun data uji (`X_test`). Ini mencegah kebocoran informasi dari data uji ke proses training dan memastikan semua fitur memiliki skala yang seragam, yang penting untuk algoritma seperti KNN dan SVM.

## Modeling

Lima algoritma klasifikasi machine learning diimplementasikan dan dilatih menggunakan data training yang telah dipersiapkan:
1.  **K-Nearest Neighbors (KNN)**: Algoritma non-parametrik yang mengklasifikasikan instance baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya. Parameter yang digunakan adalah `n_neighbors=5`.
2.  **Decision Tree (DT)**: Model berbasis pohon yang membuat keputusan berdasarkan serangkaian aturan. Parameter `random_state=42` digunakan untuk reproduktifitas.
3.  **Random Forest (RF)**: Metode ensemble yang membangun banyak decision tree. Parameter yang digunakan adalah `n_estimators=100` (default) dan `random_state=42`.
4.  **Support Vector Machine (SVM)**: Algoritma yang mencari hyperplane terbaik untuk memisahkan kelas. Menggunakan kernel default (RBF) dan `random_state=42`.
5.  **Naive Bayes (NB)**: Algoritma klasifikasi probabilistik. Menggunakan `GaussianNB`.

Setiap model dilatih pada `X_train` yang telah diskalakan dan `y_train`.

## Evaluation

Performa setiap model dievaluasi pada data uji (`X_test` dan `y_test`) menggunakan metrik-metrik berikut:
* **Confusion Matrix**: Tabel yang menunjukkan TP (True Positives), FP (False Positives), FN (False Negatives), dan TN (True Negatives).
* **Akurasi**: Persentase prediksi yang benar secara keseluruhan. Formula: `(TP + TN) / (TP + TN + FP + FN)`
* **Presisi (Precision)**: Dari semua instance yang diprediksi sebagai "Banjir", berapa banyak yang benar-benar "Banjir". Formula: `TP / (TP + FP)`
* **Recall (Sensitivity)**: Dari semua instance yang sebenarnya "Banjir", berapa banyak yang berhasil diprediksi sebagai "Banjir". Formula: `TP / (TP + FN)`
* **F1-Score**: Rata-rata harmonik dari Presisi dan Recall. Formula: `2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi Model
Berikut adalah rangkuman performa dari kelima model yang diuji (Presisi, Recall, dan F1-Score untuk kelas positif 'Banjir'):

| Model                        |   Akurasi |   Presisi |   Recall |   F1-Score |
|:-----------------------------|----------:|----------:|---------:|-----------:|
| K-Nearest Neighbors (KNN)    |     0.856 |  0.87037  | 0.810345 |   0.839286 |
| Decision Tree (DT)           |     0.88  |  0.921569 | 0.810345 |   0.862385 |
| Random Forest (RF)           |     0.888 |  0.892857 | 0.862069 |   0.877193 |
| Support Vector Machine (SVM) |     0.832 |  0.768116 | 0.913793 |   0.834646 |
| Naive Bayes (NB)             |     0.656 |  0.58427  | 0.896552 |   0.707483 |

**Deskripsi Hasil per Model:**

* **KNN Classifier:**
    * Akurasi: 85.60%.
    * TP: 47, FP: 7, FN: 11, TN: 60.
    * Model ini menunjukkan performa yang baik dengan presisi 87.04% dan recall 81.03% untuk kelas 'Banjir', menghasilkan F1-Score 83.93%.

* **Decision Tree Classifier:**
    * Akurasi: 88.00%.
    * TP: 47, FP: 4, FN: 11, TN: 63.
    * Model ini memiliki presisi yang sangat tinggi (92.16%) dan F1-Score yang baik (86.24%), meskipun recall-nya sama dengan KNN.

* **Random Forest Classifier:**
    * Akurasi: 88.80% (tertinggi).
    * TP: 50, FP: 6, FN: 8, TN: 61.
    * Menunjukkan performa terbaik secara keseluruhan dengan F1-Score tertinggi (87.72%), didukung oleh presisi (89.29%) dan recall (86.21%) yang tinggi.

* **Support Vector Machine (SVM):**
    * Akurasi: 83.20%.
    * TP: 53, FP: 16, FN: 5, TN: 51.
    * Model ini unggul dalam recall (91.38%), artinya sangat baik dalam mengidentifikasi kasus banjir aktual, namun presisinya lebih rendah (76.81%).

* **Naive Bayes Classifier:**
    * Akurasi: 65.60% (terendah).
    * TP: 52, FP: 37, FN: 6, TN: 30.
    * Meskipun recall-nya tinggi (89.66%), presisi dan akurasi keseluruhan sangat rendah, menunjukkan banyak prediksi 'Banjir' yang salah.

### Kesimpulan Pemilihan Model

Berdasarkan metrik evaluasi:
1.  **Random Forest (RF)** adalah model dengan performa paling seimbang dan akurasi tertinggi (88.8%) serta F1-Score tertinggi (87.72%). Model ini efektif dalam mengidentifikasi kejadian banjir (recall 86.21%) sambil mempertahankan presisi yang tinggi (89.29%).
2.  **Decision Tree (DT)** menunjukkan presisi tertinggi (92.16%), yang sangat berguna jika ingin meminimalkan *false alarm*.
3.  **Support Vector Machine (SVM)** memiliki recall tertinggi (91.38%), cocok jika prioritas utama adalah untuk tidak melewatkan satupun kejadian banjir, meskipun harus menerima lebih banyak *false alarm*.

Untuk kasus prediksi banjir ini, di mana penting untuk tidak melewatkan kejadian banjir (memiliki recall yang baik) namun juga tidak menimbulkan terlalu banyak alarm palsu (memiliki presisi yang baik), **Random Forest** menawarkan keseimbangan terbaik. Oleh karena itu, Random Forest dipilih sebagai model solusi terbaik dalam proyek ini.

---
**Akhir Laporan**
