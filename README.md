# Laporan Proyek Machine Learning - Muhammad Haikal

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

Dataset yang digunakan adalah "pemetaan_daerah_banjir.csv". Dataset ini bersumber dari Kaggle ([Dataset Banjir Jakarta](https://www.kaggle.com/datasets/asfilianova/dataset-banjir)) dan berisi data historis pemantauan ketinggian air serta status banjir.

### Kondisi Dataset Awal (Sebelum Pra-Pemrosesan)
Sebelum dilakukan tahapan pra-pemrosesan, dataset "pemetaan_daerah_banjir.csv" memiliki kondisi sebagai berikut:
-   **Jumlah Sampel dan Fitur**: Dataset awal terdiri dari 624 baris (sampel) dan 12 kolom (fitur).
-   **Tipe Data**: Berdasarkan output dari `data.info()` pada *notebook* (sebelum proses *cleaning*), teridentifikasi tipe data sebagai berikut:
    * `object`: 'Tanggal', 'Waktu', 'Unnamed: 10', 'Unnamed: 11'
    * `int64`: 'Katulampa', 'Pos Depok', 'Status Banjir'
    * `float64`: 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', 'Marina Ancol'
-   **Nilai Hilang (Missing Values)**: Pengecekan awal menggunakan `data.isnull().sum()` menunjukkan adanya nilai yang hilang pada beberapa kolom fitur numerik ketinggian air:
    * 'Manggarai': 1 nilai hilang (sekitar 0.16% dari total data)
    * 'Istiqlal': 2 nilai hilang (sekitar 0.32% dari total data)
    * 'Jembatan Merah': 5 nilai hilang (sekitar 0.80% dari total data)
    * 'Flusing Ancol': 1 nilai hilang (sekitar 0.16% dari total data)
    * 'Marina Ancol': 1 nilai hilang (sekitar 0.16% dari total data)
    Selain itu, kolom 'Unnamed: 10' memiliki 620 nilai hilang (sekitar 99.36%) dan 'Unnamed: 11' memiliki 621 nilai hilang (sekitar 99.52%), yang menunjukkan bahwa kolom-kolom ini sebagian besar kosong dan tidak informatif untuk seluruh dataset.


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
EDA dilakukan untuk mendapatkan pemahaman yang lebih baik tentang data (setelah pembersihan kolom tidak relevan dan penanganan nilai hilang):
1.  **Informasi Umum Data**: `data.info()` digunakan untuk melihat tipe data setiap kolom dan jumlah nilai non-null setelah pembersihan awal.
2.  **Statistik Deskriptif**: `data.describe()` digunakan untuk melihat statistik dasar fitur numerik yang akan digunakan dalam pemodelan.
3.  **Distribusi Fitur Numerik**: Visualisasi histogram untuk setiap fitur numerik ('Katulampa', 'Pos Depok', 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', 'Marina Ancol') guna memahami distribusinya.
4.  **Distribusi Variabel Target (`Status Banjir`)**: Visualisasi countplot menunjukkan distribusi kelas pada variabel target 'Status Banjir' untuk melihat apakah ada ketidakseimbangan kelas.
5.  **Heatmap Korelasi**: Untuk melihat hubungan linear antar fitur numerik yang telah dibersihkan.
6.  **Pairplot**: Untuk melihat hubungan antar pasangan fitur dan distribusi masing-masing fitur, juga dapat membantu mengidentifikasi *outlier* secara visual, meskipun dalam proyek ini *outlier* tidak ditangani secara khusus selain melalui penskalaan.

## Data Preparation
Tahapan persiapan data yang dilakukan adalah sebagai berikut (sesuai urutan implementasi pada *notebook*):
1.  **Pembersihan Kolom Tidak Relevan**: Kolom 'Unnamed: 10' dan 'Unnamed: 11' dihapus. Alasan: Kolom-kolom ini memiliki persentase nilai hilang yang sangat tinggi (lebih dari 99%) dan kontennya (seperti keterangan sumber atau catatan level siaga yang tidak konsisten) tidak akan digunakan secara langsung dalam model prediksi numerik ini.
2.  **Penghapusan Fitur Waktu**: Kolom 'Tanggal' dan 'Waktu' juga dihapus. Alasan: Meskipun informasi temporal berpotensi berguna, untuk model awal ini dan penyederhanaan, fitur ini dihilangkan agar fokus pada fitur numerik ketinggian air yang lebih langsung memengaruhi status banjir. Analisis time series tidak termasuk dalam cakupan proyek ini.
3.  **Penanganan Missing Values**: Nilai yang hilang pada kolom 'Manggarai', 'Istiqlal', 'Jembatan Merah', 'Flusing Ancol', dan 'Marina Ancol' diisi menggunakan nilai median dari masing-masing kolom. Alasan: Jumlah nilai yang hilang pada kolom-kolom ini relatif kecil (kurang dari 1% per kolom). Median dipilih sebagai strategi imputasi karena lebih robust terhadap nilai ekstrem (*outlier*) yang mungkin ada dalam data ketinggian air dibandingkan dengan mean (rata-rata).
4.  **Pemisahan Fitur dan Target**: Dataset dibagi menjadi variabel fitur (X) yang berisi data ketinggian air dari semua pos pemantauan yang tersisa, dan variabel target (y) yang berisi kolom 'Status Banjir'.
5.  **Pembagian Data (Train-Test Split)**: Data fitur (X) dan target (y) dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` dari `sklearn.model_selection`. Penggunaan `random_state=42` memastikan bahwa pembagian data dapat direproduksi. Parameter `stratify=y` digunakan untuk memastikan proporsi kelas pada variabel target ('Status Banjir') sama antara data latih dan data uji, hal ini penting untuk menghindari bias jika terdapat ketidakseimbangan kelas. **Langkah ini krusial dilakukan sebelum penskalaan fitur untuk mencegah kebocoran informasi (data leakage) dari set pengujian ke dalam proses pelatihan model.**
6.  **Penskalaan Fitur (Feature Scaling)**: Setelah pembagian data, fitur-fitur numerik dalam data latih (`X_train`) dan data uji (`X_test`) diskalakan menggunakan `MinMaxScaler` dari `sklearn.preprocessing`. Alasan: Penskalaan ini mentransformasi nilai setiap fitur ke rentang antara 0 dan 1, sehingga semua fitur memiliki skala yang seragam. Hal ini penting untuk algoritma seperti KNN dan SVM yang sensitif terhadap skala fitur. **Prosesnya adalah: *scaler* diinisialisasi, kemudian metode `fit_transform()` diterapkan *hanya* pada data latih (`X_train`) untuk mempelajari parameter skala (nilai minimum dan maksimum dari setiap fitur dalam data latih). Setelah itu, metode `transform()` digunakan untuk menerapkan penskalaan tersebut baik pada `X_train` maupun pada `X_test` menggunakan parameter yang telah dipelajari dari `X_train`.** Pendekatan ini memastikan bahwa data uji tetap "tak terlihat" selama proses pembelajaran parameter skala, sehingga evaluasi model pada data uji menjadi lebih valid dan mencerminkan bagaimana model akan berperforma pada data baru yang belum pernah dilihat sebelumnya.

## Modeling

Lima algoritma klasifikasi machine learning diimplementasikan dan dilatih menggunakan data training (`X_train` yang telah diskalakan dan `y_train`) yang telah dipersiapkan:

1.  **K-Nearest Neighbors (KNN)**:
    * **Mekanisme Kerja**: KNN adalah algoritma pembelajaran berbasis instans yang mengklasifikasikan sampel baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya dalam ruang fitur. Jarak antar sampel biasanya dihitung menggunakan metrik jarak Euclidean. Saat prediksi, KNN akan mencari 'k' sampel dari data latih yang paling mirip (jarak terdekat) dengan sampel baru, kemudian kelas mayoritas dari 'k' tetangga tersebut akan menjadi prediksi kelas untuk sampel baru.
    * **Parameter Utama**: `n_neighbors=5`. Parameter `n_neighbors` (k) menentukan jumlah tetangga yang dipertimbangkan. Pemilihan nilai 'k' bersifat krusial; nilai 'k' yang terlalu kecil dapat membuat model sensitif terhadap *noise* dan *overfitting*, sedangkan nilai 'k' yang terlalu besar dapat menghaluskan batas keputusan secara berlebihan dan menyebabkan *underfitting*. Nilai 5 dipilih sebagai nilai awal yang umum digunakan dan sering memberikan keseimbangan yang baik antara bias dan varians untuk banyak dataset.

2.  **Decision Tree (DT)**:
    * **Mekanisme Kerja**: Decision Tree membangun model klasifikasi atau regresi dalam bentuk struktur pohon. Algoritma ini bekerja dengan cara mempartisi ruang fitur secara rekursif menjadi subset-subset yang lebih kecil berdasarkan nilai fitur tertentu pada setiap *node* keputusan. Proses pemecahan ini berlanjut hingga mencapai *node* daun (*leaf node*) yang merepresentasikan label kelas atau nilai prediksi, atau hingga kriteria pemberhentian terpenuhi (misalnya, kedalaman maksimum pohon atau jumlah minimum sampel pada *leaf node*). Pemisahan pada setiap *node* dilakukan dengan mencari fitur dan ambang batas yang paling baik dalam memisahkan kelas, biasanya menggunakan metrik seperti Gini impurity atau information gain untuk memaksimalkan kemurnian kelas pada *node* anak.
    * **Parameter Utama**: `random_state=42`. Parameter ini digunakan untuk mengontrol keacakan dalam proses pembentukan pohon (misalnya, dalam pemilihan fitur jika beberapa fitur memiliki *gain* yang sama atau dalam pemecahan pada *node* jika ada beberapa pemecahan yang sama baiknya). Pengaturan `random_state` memastikan bahwa hasil pembentukan pohon dan pelatihan model dapat direproduksi setiap kali kode dijalankan. Parameter lain seperti kedalaman maksimum pohon (`max_depth`) atau jumlah minimum sampel untuk melakukan pemisahan (`min_samples_split`) tidak diatur secara spesifik dalam implementasi ini, sehingga menggunakan nilai default dari pustaka `sklearn`.

3.  **Random Forest (RF)**:
    * **Mekanisme Kerja**: Random Forest adalah metode *ensemble learning* yang bekerja dengan membangun sejumlah besar Decision Tree secara acak selama proses pelatihan. Setiap pohon dalam *forest* dilatih pada subset acak dari data latih (menggunakan teknik *bootstrap sampling*, yaitu pengambilan sampel dengan penggantian dari data latih asli) dan pada setiap pemisahan *node*, hanya subset acak dari fitur yang dipertimbangkan untuk menentukan pemisahan terbaik (ini menambah keragaman). Untuk klasifikasi, prediksi akhir ditentukan oleh mayoritas suara (voting) dari semua pohon individu. Pendekatan ini membantu mengurangi *overfitting* yang sering terjadi pada Decision Tree tunggal dan meningkatkan generalisasi model dengan mengurangi varians.
    * **Parameter Utama**: `n_estimators=100` dan `random_state=42`. `n_estimators` menentukan jumlah pohon yang akan dibangun dalam *forest*; nilai 100 adalah nilai default yang sering memberikan performa yang baik tanpa menjadi terlalu berat secara komputasi. Semakin banyak pohon cenderung meningkatkan performa hingga titik tertentu, namun juga meningkatkan waktu pelatihan. `random_state=42` memastikan bahwa proses pembentukan *forest* yang melibatkan keacakan (seperti *bootstrap sampling* dan pemilihan fitur acak pada setiap *split*) dapat direproduksi, sehingga hasil pelatihan model konsisten.

4.  **Support Vector Machine (SVM)**:
    * **Mekanisme Kerja**: SVM adalah algoritma klasifikasi yang bertujuan untuk menemukan *hyperplane* (batas keputusan) terbaik yang memisahkan dua kelas data dalam ruang fitur dengan margin terbesar. Margin didefinisikan sebagai jarak antara *hyperplane* dan titik data terdekat dari masing-masing kelas (titik-titik ini disebut *support vectors*). Dengan memaksimalkan margin, SVM berusaha mencapai generalisasi yang baik dan robust terhadap *outlier*. Untuk kasus data yang tidak dapat dipisahkan secara linear dalam ruang fitur aslinya, SVM dapat menggunakan fungsi kernel (seperti RBF, polinomial, atau sigmoid) untuk memetakan data ke ruang dimensi yang lebih tinggi di mana pemisahan linear mungkin dapat dilakukan.
    * **Parameter Utama**: Menggunakan kernel default yaitu RBF (Radial Basis Function) dan `random_state=42`. Kernel RBF adalah pilihan umum karena fleksibilitasnya dalam menangani hubungan non-linear antar fitur. Parameter `C` (parameter regularisasi) dan `gamma` (parameter kernel) tidak diatur secara spesifik, sehingga menggunakan nilai default. `random_state` digunakan jika ada aspek keacakan dalam proses optimasi internal algoritma SVM di `sklearn` (meskipun SVM pada dasarnya deterministik untuk kernel tertentu, beberapa optimasi mungkin memiliki komponen acak).

5.  **Naive Bayes (NB)**:
    * **Mekanisme Kerja**: Naive Bayes adalah keluarga algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes. Disebut "naive" karena membuat asumsi (yang seringkali merupakan penyederhanaan kuat) bahwa semua fitur adalah independen satu sama lain, mengingat kelas variabel. `GaussianNB` adalah salah satu varian yang digunakan ketika fitur-fitur bersifat kontinu dan diasumsikan terdistribusi secara Gaussian (normal) dalam setiap kelas. Algoritma ini menghitung probabilitas posterior untuk setiap kelas ($P(\text{kelas} | \text{fitur})$) berdasarkan probabilitas prior kelas ($P(\text{kelas})$) dan probabilitas kondisional dari fitur-fitur ($P(\text{fitur} | \text{kelas})$), kemudian memilih kelas dengan probabilitas posterior tertinggi sebagai prediksi.
    * **Parameter Utama**: Menggunakan `GaussianNB` karena fitur-fitur dalam dataset ini (ketinggian air) adalah numerik kontinu. `GaussianNB` tidak memiliki banyak parameter utama yang perlu di-tuning secara manual untuk kasus *baseline*; ia mempelajari parameter distribusi Gaussian (mean dan standar deviasi) dari data latih untuk setiap fitur per kelas.

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
