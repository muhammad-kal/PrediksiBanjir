# Laporan Proyek Machine Learning - Muhammad Haikal

## Domain Proyek

Banjir merupakan salah satu bencana alam yang sering terjadi di berbagai wilayah, termasuk Indonesia, dan dapat menyebabkan kerugian signifikan baik secara material maupun non-material. Keterlambatan dalam mendeteksi potensi banjir seringkali memperparah dampak yang ditimbulkan. Oleh karena itu, pengembangan sistem prediksi banjir yang akurat dan tepat waktu menjadi sangat penting. Dengan memanfaatkan data historis ketinggian air dari berbagai pos pemantauan, kita dapat membangun model machine learning yang mampu memberikan peringatan dini akan potensi terjadinya banjir. Proyek ini bertujuan untuk membuat model predictive analytics untuk memprediksi status banjir berdasarkan data-data tersebut.

Dataset yang digunakan dalam proyek ini bersumber dari Kaggle: [Dataset Banjir Jakarta](https://www.kaggle.com/datasets/asfilianova/dataset-banjir).

## Business Understanding

### Problem Statements
1.  Bagaimana cara membangun model machine learning yang dapat memprediksi status potensi banjir (Banjir/Tidak Banjir) secara akurat berdasarkan data historis ketinggian air dari berbagai pos pemantauan?
2.  Algoritma machine learning manakah yang memberikan performa terbaik dalam melakukan klasifikasi status banjir pada dataset yang digunakan?
3.  Bagaimana mengevaluasi dan membandingkan performa dari berbagai model machine learning untuk kasus prediksi banjir ini?

### Goals
1.  Mengembangkan model prediksi yang mampu mengklasifikasikan status banjir dengan tingkat akurasi, presisi, dan recall yang baik.
2.  Melakukan perbandingan performa antara beberapa algoritma machine learning (K-Nearest Neighbors, Decision Tree, Random Forest, SVM, dan Naive Bayes) untuk menemukan model yang paling optimal.
3.  Menyediakan dasar untuk sistem peringatan dini banjir yang dapat membantu dalam pengambilan keputusan dan mitigasi bencana.

### Solution Statements
Untuk mencapai tujuan tersebut, solusi yang diajukan adalah sebagai berikut:
1.  Melakukan pra-pemrosesan data secara komprehensif yang mencakup pembersihan data, penanganan nilai yang hilang, dan transformasi fitur yang diperlukan.
2.  Menerapkan dan melatih beberapa algoritma klasifikasi machine learning pada data yang telah diproses.
3.  Mengevaluasi setiap model menggunakan metrik performa yang relevan (Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix) untuk memilih model terbaik sebagai solusi prediksi banjir.

## Data Understanding

Dataset yang digunakan ("pemetaan\_daerah\_banjir.csv") berisi data historis pemantauan ketinggian air dan status banjir.
-   Jumlah sampel data awal: 624 baris
-   Jumlah fitur awal: 12 kolom

Berikut adalah deskripsi singkat mengenai fitur-fitur utama dalam dataset setelah pembersihan awal (penghapusan kolom 'Unnamed: 10' dan 'Unnamed: 11'):
* `Tanggal`: Tanggal pencatatan data (tipe object).
* `Waktu`: Waktu pencatatan data per jam (tipe object).
* `Katulampa`: Ketinggian air di pos pemantauan Katulampa (cm, tipe int64).
* `Pos Depok`: Ketinggian air di pos pemantauan Depok (cm, tipe int64).
* `Manggarai`: Ketinggian air di pos pemantauan Manggarai (cm, tipe float64).
* `Istiqlal`: Ketinggian air di pos pemantauan Istiqlal (cm, tipe float64).
* `Jembatan Merah`: Ketinggian air di pos pemantauan Jembatan Merah (cm, tipe float64).
* `Flusing Ancol`: Ketinggian air di pos pemantauan Flusing Ancol (cm, tipe float64).
* `Marina Ancol`: Ketinggian air di pos pemantauan Marina Ancol (cm, tipe float64).
* `Status Banjir`: Variabel target yang menunjukkan status banjir; 0 berarti Tidak Banjir, 1 berarti Banjir (tipe int64).

### Exploratory Data Analysis (EDA)
EDA dilakukan untuk mendapatkan pemahaman yang lebih baik tentang data:
1.  **Informasi Umum Data**: Melihat tipe data setiap kolom dan jumlah nilai non-null.
2.  **Statistik Deskriptif**: Menggunakan `describe()` untuk melihat statistik dasar fitur numerik (mean, std, min, max, kuartil).
3.  **Distribusi Fitur Numerik**: Visualisasi histogram untuk setiap fitur numerik guna memahami distribusinya.
    4.  **Distribusi Variabel Target (`Status Banjir`)**: Visualisasi countplot untuk melihat proporsi kelas pada variabel target. Diketahui bahwa terdapat ketidakseimbangan kelas, meskipun tidak ekstrem.
    5.  **Heatmap Korelasi**: Untuk melihat hubungan linear antar fitur numerik. Ini membantu mengidentifikasi potensi multikolinearitas.
    Beberapa kolom seperti 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', dan 'Marina Ancol' teridentifikasi memiliki nilai yang hilang (missing values).

## Data Preparation

Tahapan persiapan data yang dilakukan adalah sebagai berikut:
1.  **Pembersihan Kolom Tidak Relevan**: Kolom 'Unnamed: 10' dan 'Unnamed: 11' dihapus karena sebagian besar nilainya kosong dan tidak relevan untuk pemodelan.
2.  **Penanganan Missing Values**: Nilai yang hilang pada kolom 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', dan 'Marina Ancol' diisi menggunakan nilai median dari masing-masing kolom. Median dipilih karena lebih robust terhadap outlier dibandingkan mean, yang mungkin ada pada data ketinggian air.
3.  **Penghapusan Fitur Waktu**: Kolom 'Tanggal' dan 'Waktu' dihapus. Meskipun berpotensi memberikan informasi temporal, untuk penyederhanaan model awal dan menghindari kompleksitas feature engineering lanjutan, kedua kolom ini tidak disertakan dalam fitur prediktor.
4.  **Pemisahan Fitur dan Target**: Dataset dibagi menjadi fitur (X) dan variabel target (y = 'Status Banjir').
5.  **Pembagian Data (Train-Test Split)**: Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari `sklearn.model_selection`. Penggunaan `random_state=42` memastikan reproduktifitas hasil pembagian, dan `stratify=y` digunakan untuk menjaga proporsi kelas target yang sama pada data latih dan data uji, mengingat adanya sedikit ketidakseimbangan kelas.
6.  **Penskalaan Fitur (Feature Scaling)**: Fitur-fitur numerik dalam data latih dan data uji diskalakan menggunakan `StandardScaler`. Penskalaan ini bertujuan untuk menstandarisasi rentang nilai fitur sehingga memiliki mean 0 dan standar deviasi 1. Scaler di-'fit' *hanya* pada data latih (`X_train`) dan kemudian digunakan untuk mentransformasi baik data latih maupun data uji (`X_test`) untuk mencegah kebocoran informasi dari data uji.

## Modeling

Lima algoritma klasifikasi machine learning diimplementasikan dan dilatih menggunakan data training yang telah dipersiapkan:
1.  **K-Nearest Neighbors (KNN)**: Algoritma non-parametrik yang mengklasifikasikan instance baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya dalam ruang fitur. Parameter `n_neighbors` diatur ke 5.
2.  **Decision Tree (DT)**: Model berbasis pohon yang membuat keputusan berdasarkan serangkaian aturan if-then-else yang dipelajari dari fitur data. Parameter `random_state=42` digunakan untuk reproduktifitas.
3.  **Random Forest (RF)**: Metode ensemble learning yang membangun banyak decision tree selama training dan mengeluarkan kelas yang merupakan modus dari kelas yang dikeluarkan oleh masing-masing tree. Parameter `n_estimators=100` dan `random_state=42` digunakan.
4.  **Support Vector Machine (SVM)**: Algoritma yang mencari hyperplane terbaik yang memisahkan dua kelas dalam ruang fitur. Menggunakan kernel default (RBF) dan `random_state=42`.
5.  **Naive Bayes (NB)**: Algoritma klasifikasi probabilistik berdasarkan teorema Bayes dengan asumsi independensi antar fitur. Menggunakan `GaussianNB` yang cocok untuk fitur kontinu.

Setiap model dilatih pada `X_train` yang telah diskalakan dan `y_train`.

## Evaluation

Performa setiap model dievaluasi pada data uji (`X_test` dan `y_test`) menggunakan metrik-metrik berikut:
* **Confusion Matrix**: Tabel yang merangkum hasil prediksi klasifikasi, menunjukkan True Positives (TP), False Positives (FP), False Negatives (FN), dan True Negatives (TN).
    * TP (Banjir): Jumlah kasus banjir yang diprediksi benar sebagai banjir.
    * FP (Banjir): Jumlah kasus tidak banjir yang salah diprediksi sebagai banjir (False Alarm).
    * FN (Banjir): Jumlah kasus banjir yang salah diprediksi sebagai tidak banjir (Missed Event).
    * TN (Banjir): Jumlah kasus tidak banjir yang diprediksi benar sebagai tidak banjir.
* **Akurasi**: Persentase total prediksi yang benar dari keseluruhan data uji.
    Formula: `(TP + TN) / (TP + TN + FP + FN)`
* **Presisi (Precision)**: Dari semua instance yang diprediksi sebagai kelas positif (Banjir), berapa banyak yang benar-benar positif. Berguna untuk mengetahui seberapa bisa diandalkan prediksi positif dari model.
    Formula: `TP / (TP + FP)`
* **Recall (Sensitivity)**: Dari semua instance yang sebenarnya positif (Banjir), berapa banyak yang berhasil diprediksi sebagai positif oleh model. Berguna untuk mengetahui seberapa baik model menemukan semua kasus positif.
    Formula: `TP / (TP + FN)`
* **F1-Score**: Rata-rata harmonik dari Presisi dan Recall, memberikan ukuran tunggal yang menyeimbangkan kedua metrik tersebut.
    Formula: `2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi Model
Berikut adalah rangkuman performa dari kelima model yang diuji:

| Model                        |   Akurasi |   Presisi |   Recall |   F1-Score |
|:-----------------------------|----------:|----------:|---------:|-----------:|
| K-Nearest Neighbors (KNN)    |     0.856 |  0.87037  | 0.810345 |   0.839286 |
| Decision Tree (DT)           |     0.88  |  0.921569 | 0.810345 |   0.862385 |
| Random Forest (RF)           |     0.888 |  0.892857 | 0.862069 |   0.877193 |
| Support Vector Machine (SVM) |     0.832 |  0.768116 | 0.913793 |   0.834646 |
| Naive Bayes (NB)             |     0.656 |  0.58427  | 0.896552 |   0.707483 |

*(Nilai Presisi, Recall, dan F1-Score di atas adalah untuk kelas positif, yaitu 'Banjir')*

### Analisis Hasil
* **Random Forest (RF)** menunjukkan performa keseluruhan terbaik dengan Akurasi tertinggi (88.8%) dan F1-Score tertinggi (87.7%). Model ini juga memiliki Presisi (89.3%) dan Recall (86.2%) yang sangat baik, menunjukkan keseimbangan yang baik antara meminimalkan false alarm dan tidak melewatkan kejadian banjir.
* **Decision Tree (DT)** memiliki Presisi tertinggi (92.2%), artinya sangat andal ketika memprediksi "Banjir", namun Recall-nya (81.0%) sedikit lebih rendah dibandingkan RF. Akurasinya juga sangat baik (88.0%).
* **K-Nearest Neighbors (KNN)** memberikan performa yang solid dengan akurasi 85.6% dan F1-Score 83.9%.
* **Support Vector Machine (SVM)** memiliki Recall tertinggi kedua (91.4%), menunjukkan kemampuan yang sangat baik untuk mendeteksi kasus banjir aktual. Namun, Presisinya lebih rendah (76.8%), yang berarti menghasilkan lebih banyak false alarm.
* **Naive Bayes (NB)** juga memiliki Recall yang tinggi (89.7%), tetapi Akurasi (65.6%) dan Presisinya (58.4%) paling rendah, menjadikannya model yang kurang dapat diandalkan secara keseluruhan untuk kasus ini.

### Kesimpulan Pemilihan Model

Berdasarkan evaluasi yang komprehensif:
1.  **Random Forest (RF)** direkomendasikan sebagai model terbaik untuk prediksi status banjir pada dataset ini. Alasannya adalah Akurasi dan F1-Score tertinggi, serta keseimbangan yang sangat baik antara Presisi dan Recall. Ini menunjukkan model yang tidak hanya akurat secara keseluruhan tetapi juga efektif dalam mengidentifikasi kasus banjir sambil menjaga tingkat false alarm tetap rendah.
2.  Jika prioritas utama adalah untuk meminimalkan prediksi banjir yang salah (false alarm), **Decision Tree (DT)** bisa menjadi alternatif yang kuat karena presisinya yang paling tinggi.
3.  Jika prioritasnya adalah memastikan seminimal mungkin kejadian banjir yang terlewat (meminimalkan false negative), **SVM** menunjukkan recall yang sangat tinggi, meskipun dengan presisi yang lebih rendah.

Dengan demikian, untuk solusi yang paling seimbang dan berkinerja tinggi, **Random Forest** adalah pilihan yang paling tepat.

---
**Akhir Laporan**
