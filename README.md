# *Prediksi Penyakit Diabetes Menggunakan Machine Learning*
---
## Domain Proyek

Diabetes merupakan penyakit kronis yang memengaruhi jutaan orang di seluruh dunia. Penyakit ini ditandai dengan tingginya kadar gula darah (glukosa) dalam tubuh. Kondisi ini terjadi karena tubuh tidak mampu memproduksi atau menggunakan hormon insulin secara efektif, sehingga glukosa tidak dapat digunakan sebagai sumber energi oleh sel-sel tubuh. Deteksi dini penyakit diabetes sangat penting untuk mencegah komplikasi yang lebih serius dan dapat dilakukan penanganan awal. Oleh karena itu, pengembangan model prediksi diabetes berbasis machine learning dapat membantu dalam diagnosis awal dan langkah yang diperlukan setelahnya.

Referensi:  
- [American Diabetes Association (2020). Classification and Diagnosis of Diabetes: Standards of Medical Care in Diabetes—2020.](https://doi.org/10.2337/dc20-S002)
- [D. Sisodia and D. S. Sisodia, “Prediction of Diabetes using Classification Algorithms,” Procedia Comput. Sci., vol. 132, pp. 1578–1585, 2018, doi: 10.1016/j.procs.2018.05.122.](https://doi.org/10.32520/stmsi.v10i1.1129)
---
## Business Understanding

### Problem Statements
- Bagaimana memprediksi apakah seseorang berisiko diabetes berdasarkan data medisnya?
- Bagaimana memanfaatkan model prediksi tersebut untuk membantu tenaga medis dalam pengambilan keputusan?

### Goals
- Membuat model klasifikasi yang dapat memprediksi risiko diabetes dengan akurasi tinggi.
- Membandingkan performa beberapa algoritma untuk menentukan model terbaik.

### Solution Statements
- Membangun baseline model menggunakan Logistic Regression dan Random Forest.
- Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa.

---
## Data Understanding

#### Deskripsi Dataset

Dataset ini berisi data mengenai pasien yang memiliki atau berisiko mengalami diabetes. Setiap baris mewakili satu pasien, dan terdapat berbagai fitur medis yang diukur untuk setiap individu.
Dataset ini mencakup **768 entri** data dengan **12 kolom** fitur. Dataset ini memiliki label kelas yang mengindikasikan status diabetes pasien, yaitu **Non-diabetic (N)**, **Prediabetic (P)**, atau **Diabetic (Y)**.

#### Sumber Dataset
Dataset ini dapat ditemukan di [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset/data?select=Dataset+of+Diabetes+.csv).

#### Variabel pada Dataset

1. **ID**: Identifikasi unik untuk setiap record pasien.
2. **No_Pation**: Nomor identifikasi pasien (mungkin ID rekam medis atau nomor pasien).
3. **Gender**: Jenis kelamin pasien:
   - `F`: Perempuan
   - `M`: Laki-laki

4. **AGE**: Usia pasien dalam tahun.
5. **Urea**: Tingkat urea dalam darah pasien (mg/dL atau mmol/L). Urea adalah produk sampingan metabolisme protein dan dapat memberikan indikasi fungsi ginjal.
6. **Cr**: Tingkat kreatinin dalam darah (mg/dL atau µmol/L). Kreatinin juga merupakan indikator fungsi ginjal.
7. **HbA1c**: Hemoglobin terglikasi, yang menunjukkan rata-rata kadar gula darah pasien selama 2-3 bulan terakhir (dalam bentuk persentase).
8. **Chol**: Kadar kolesterol total dalam darah (mg/dL atau mmol/L).
9. **TG**: Kadar trigliserida dalam darah (mg/dL atau mmol/L). Trigliserida adalah jenis lemak dalam darah.
10. **HDL**: Kadar kolesterol high-density lipoprotein (HDL), sering disebut sebagai kolesterol "baik" (mg/dL atau mmol/L).
11. **LDL**: Kadar kolesterol low-density lipoprotein (LDL), sering disebut sebagai kolesterol "jahat" (mg/dL atau mmol/L).
12. **VLDL**: Kadar kolesterol very low-density lipoprotein (VLDL), yang juga merupakan jenis kolesterol (mg/dL atau mmol/L).
13. **BMI**: Indeks massa tubuh (BMI), yang mengukur proporsi lemak tubuh berdasarkan tinggi badan dan berat badan (kg/m²).
14. **CLASS**: Label status diabetes dari pasien:
    - `N`: Non-diabetic
    - `P`: Prediabetic
    - `Y`: Diabetic

---
## EDA (Exploratory Data Analysist)

### 1. Deskrisi Statistik
![Deskripsi Statistik](img\describe.jpg)

dari fungsi `describe()` kita dapat mengetahui beberapa informasi :

- **AGE**: Usia pasien rata-rata 53.5 tahun, dengan variasi antara 20 hingga 79 tahun.
- **Urea** dan **Cr (Kreatinin)**: Kadar urea dan kreatinin menunjukkan rentang yang cukup besar, yang dapat memberikan indikasi mengenai fungsi ginjal pasien.
- **HbA1c**: Rata-rata kadar HbA1c sebesar 8.28% menunjukkan bahwa banyak pasien memiliki kadar gula darah lebih tinggi dari batas normal (6.5%).
- **Kolesterol (Chol, TG, HDL, LDL, VLDL)**: Kolesterol total, trigliserida, HDL, LDL, dan VLDL menunjukkan variasi yang cukup besar, yang dapat mempengaruhi kesehatan jantung.
- **BMI**: Rata-rata BMI adalah 29.58, yang menunjukkan bahwa sebagian besar pasien termasuk dalam kategori overweight atau obesitas.

### 2. Boxplot Data Numerik
![Boxplot Numerik](img\boxplot_numerik.png)

Boxplot yang ditampilkan menunjukkan beberapa fitur dalam dataset yang memiliki outliers yang terdeteksi. Outliers ini ditunjukkan dengan titik di luar whiskers pada setiap boxplot. Berikut adalah penjelasan mengenai fitur-fitur yang memiliki outliers:

1. **Boxplot Gender**:
   - Kolom **Gender** tidak menunjukkan adanya outliers karena hanya berisi dua kategori (`M` dan `F`).
   - Oleh karena itu, tidak ada distribusi yang perlu dianalisis lebih lanjut pada kolom ini.

2. **Boxplot Urea dan Boxplot Cr**:
   - **Urea** dan **Creatinine (Cr)** menunjukkan adanya nilai yang sangat tinggi yang berada di luar whiskers.
   - Nilai outliers pada kolom ini bisa mengindikasikan masalah pada fungsi ginjal, namun perlu dianalisis lebih lanjut apakah ini merupakan data yang sah atau kesalahan dalam pencatatan.

3. **Boxplot HbA1c**:
   - Kolom **HbA1c** menunjukkan adanya outliers dengan nilai yang sangat tinggi, yang menunjukkan bahwa beberapa pasien mungkin memiliki kadar gula darah yang sangat tidak terkontrol.
   - Pasien dengan nilai HbA1c tinggi mungkin menunjukkan kondisi diabetes yang buruk dan membutuhkan perhatian medis lebih lanjut.

4. **Boxplot Chol, TG, HDL, LDL, VLDL**:
   - Kolom **Cholesterol (Chol)**, **Triglycerides (TG)**, **HDL**, **LDL**, dan **VLDL** menunjukkan adanya outliers yang dapat dikaitkan dengan masalah kesehatan kardiovaskular.
   - Outliers pada kadar kolesterol dan trigliserida dapat menunjukkan pasien dengan risiko penyakit jantung atau gangguan metabolik, yang merupakan indikasi penting dalam diagnosis dan perawatan.

Kesimpulan:
- Outliers pada **AGE**, **Urea**, **Creatinine**, **HbA1c**, dan kolesterol menunjukkan adanya variasi yang signifikan dalam kondisi medis pasien.
- Beberapa nilai outliers mungkin merupakan data yang sah, misalnya pada pasien dengan gangguan ginjal atau diabetes yang tidak terkontrol.
- Pada dataset ini kita asumsikan bahwa data tersebut memang apa adanya karena dalam dunia medis sering ditemukan outliers pada beberapa fitur

### 3. Distribusi Gender

![Distribusi Gender](img\distribusi_gender.png)

pada kolom `gender` menunjukkan perbandingan yang hampir seimbang yaitu untuk **Male** sebanyak **565** data dan untuk **Female** sebanyak **435**. Namun terdapat kesalahan penulisan pada salah satu data yaitu menggunakan huruf kecil pada `f` yang mana ini menunjukkan inkonsistensi. Maka perlu dirubah agar data menjadi konsisten

### 4. Distribusi CLASS

![Distribusi CLASS](img\distribusi_label.png)

pada kolom `CLASS` menunjukkan perbandingan yang yang tidak seimbang / imbalance yaitu untuk **Y** sebanyak **844** data, untuk **N** sebanyak **103**,dan untuk **P** sebanyak **53**. Namun terdapat beberapa data yang terjadi kesalahan penulisan yaitu `Y`  dan `N` ,yang mana ini menunjukkan inkonsistensi. Maka perlu dirubah agar data menjadi konsisten

### 4. Distibusi Numerik (Histogram)

![Distribusi Numerik](img\distribusi_numerik.png)

 Penjelasan Distribusi Histogram untuk Fitur Numerik

1. **AGE**:
   - Distribusi **AGE** cenderung memiliki dua puncak, dengan konsentrasi utama pada usia sekitar 40-50 tahun. Ini menunjukkan bahwa sebagian besar pasien berada pada kelompok usia menengah.
   - Ada beberapa pasien di usia yang lebih tua, tetapi secara umum, distribusinya lebih terpusat pada rentang usia menengah.

2. **Urea**:
   - Distribusi **Urea** menunjukkan konsentrasi yang lebih tinggi pada nilai yang lebih rendah, dengan banyak pasien memiliki kadar urea di sekitar 5 mg/dL. 
   - Ada beberapa outliers dengan nilai urea yang sangat tinggi, yang bisa menunjukkan gangguan fungsi ginjal.

3. **Cr (Kreatinin)**:
   - Distribusi **Cr** menunjukkan puncak yang sangat tinggi di sekitar 100, dengan sebagian besar data terkonsentrasi di bawah 200 mg/dL.
   - Ada beberapa pasien dengan kadar kreatinin sangat tinggi, yang menunjukkan adanya masalah pada fungsi ginjal.

4. **HbA1c**:
   - **HbA1c** memiliki distribusi yang miring ke kanan, dengan banyak pasien memiliki nilai di bawah 8%. 
   - Ini menunjukkan bahwa sebagian besar pasien berada dalam rentang prediabetes atau diabetes yang tidak terkontrol dengan kadar HbA1c yang lebih tinggi.

5. **Chol (Kolesterol Total)**:
   - Distribusi **Chol** cenderung normal, dengan puncak di sekitar 4-6 mg/dL. Namun, ada beberapa outliers dengan kadar kolesterol yang sangat tinggi.
   - Ini menunjukkan adanya variasi yang cukup besar dalam kadar kolesterol darah pasien.

6. **TG (Trigliserida)**:
   - Distribusi **TG** memiliki puncak di sekitar 2-3 mg/dL, dengan beberapa pasien memiliki kadar trigliserida yang sangat tinggi (outliers).
   - Peningkatan kadar trigliserida dapat mengindikasikan risiko penyakit kardiovaskular.

7. **HDL (Kolesterol HDL)**:
   - Distribusi **HDL** menunjukkan konsentrasi di bawah 2, dengan beberapa pasien memiliki kadar HDL yang lebih tinggi.
   - HDL adalah kolesterol "baik," dan sebagian besar pasien tampaknya memiliki kadar yang lebih rendah dari yang diinginkan (idealnya di atas 2 mg/dL).

8. **LDL (Kolesterol LDL)**:
   - Distribusi **LDL** menunjukkan konsentrasi yang lebih tinggi pada nilai rendah (sekitar 2-3 mg/dL), dengan beberapa outliers yang menunjukkan kadar LDL sangat tinggi.
   - Ini menunjukkan adanya risiko yang lebih tinggi terkait dengan kolesterol "jahat" dalam darah.

9. **VLDL**:
   - Distribusi **VLDL** menunjukkan puncak di sekitar nilai yang sangat rendah, dengan beberapa outliers pada nilai tinggi.
   - VLDL adalah jenis kolesterol yang berbahaya, dan distribusinya menunjukkan sebagian besar pasien memiliki kadar yang lebih rendah.

10. **BMI**:
   - Distribusi **BMI** menunjukkan bahwa sebagian besar pasien memiliki BMI di sekitar 25-30, yang menunjukkan kelompok pasien dengan kategori overweight atau obesitas.
   - Ada banyak variasi pada BMI, dengan beberapa pasien memiliki BMI yang lebih tinggi dari nilai normal (30 ke atas).

 Kesimpulan:
- **AGE**, **Urea**, **Cr**, **HbA1c**, dan **Chol** menunjukkan variasi yang cukup besar di antara pasien. Beberapa fitur, seperti **Urea**, **Cr**, dan **HbA1c**, menunjukkan puncak yang kuat, yang bisa mengindikasikan kondisi medis yang lebih serius pada sebagian pasien.
- Kolesterol dan trigliserida (**Chol**, **TG**, **HDL**, **LDL**, **VLDL**) menunjukkan variasi yang normal, tetapi dengan beberapa nilai ekstrem yang mungkin mencerminkan risiko kardiovaskular pada beberapa pasien.
- **BMI** menunjukkan bahwa sebagian besar pasien berada dalam kategori overweight atau obesitas, yang merupakan faktor risiko untuk diabetes.


## Data Preparation

### 1: Menangani Data Inkonsisten 

![Inkonsisten Data](img\inkonsisten_data_gender.jpg)
![Inkonsisten Data](img\inkonsisten_data_label.jpg)



```
df['Gender'] = df['Gender'].replace({'f': 'F'})
df['Gender'].value_counts()

```

fungsi `replace()` pada `df['Gender'].replace({'f': 'F'}` merubah karakter 'f' menjadi 'F' agar sesuai dengan nilai utama

```

df['CLASS'] = df['CLASS'].replace({'Y ': 'Y', 'N ': 'N'})
df['CLASS'].value_counts()

```
fungsi `replace()` pada `df['CLASS'].replace({'Y ': 'Y', 'N ': 'N'})` merubah karakter 'Y ' dan 'N ' menjadi 'Y' dan 'N' agar sesuai dengan nilai utama

### 2. Encoding Categorical

![Encoding](img\encode_categorical.jpg)

`LabelEncoder()` digunakan untuk merubah nilai kategorikal menjadi bentuk dari **0 sampai banyaknya data -1**, karena pada *Gender* memiliki 2 nilai yaitu F dan M , maka dirubah menjadi F = 0 , M = 1. Sedangkan pada *CLASS* memiliki 3 nilai yaitu Y, N, dan P maka dirubah menjadi N= 0, P= 1, Y= 2 

### 3. Correlation Matrix setelah Encode

![Correlation Matrix](img\corr_metrik.png)

Berdasarkan hasil **korelasi matrix** , fitur yang memiliki keterkaitan yang kuat dengan target `CLASS` yaitu `BMI`, `HbA1C`, dan `AGE` . sedangkan kolom `VLDL`, `TG`, `Chol`, dan `Gender` memiliki keterkaitan dengan target namun lemah.

### 4. SMOTE (Oversampling)

![SMOTE](img\smote.png)

Hasil dari `SMOTE` dapat dilihat yaitu sekarang target menunjukkan sudah balance dengan masing masing memiliki nilai **671**

### 5. Standarisasi

```

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

```

`StandardScaler` digunakan untuk menstandarisasi data agar fitur memiliki skala yang sama, dan ini diterapkan pada data latih yang sudah dioversampling dan data uji.

## Modelling

### 1. Logistic Regression

![Logistic Regression](img\lr_matrix.jpg)
![Logistic Regression](img\lr_confusion.png)

pada model `Random Forest` hasil akurasi didapatkan mencapai **90%** dengan masing masing :

- pada **class** 0 
  
  `presisi` : 88%

  `recall`  : 100%

  `f1-score` : 93%

- pada **class** 1 
  
  `presisi` : 100%

  `recall`  : 100%

  `f1-score` : 100%

- pada **class** 2 
  
  `presisi` : 100%

  `recall`  : 98%

  `f1-score` : 99%


Hasil ini menunjukkan bahwa model ini sudah cukup baik untuk mengenali pola dari masing-masing nilai pada target.

### 2. Random Forest

![Random Forest](img\rf_matrix.jpg)
![Random Forest](img\rf_confusion.png)

pada model `Random Forest` hasil akurasi didapatkan mencapai **90%** dengan masing masing :

- pada **class** 0 
  
  `presisi` : 88%

  `recall`  : 100%

  `f1-score` : 93%

- pada **class** 1 
  
  `presisi` : 100%

  `recall`  : 100%

  `f1-score` : 100%

- pada **class** 2 
  
  `presisi` : 100%

  `recall`  : 98%

  `f1-score` : 99%


Hasil ini menunjukkan bahwa model ini sudah cukup baik untuk mengenali pola dari masing-masing nilai pada target.

## Evaluasi

![Evaluasi](img\evaluasi.jpg)

perbandingan antara `Logistic Regression` dan `Random Forest` menunjukkan bahwa `Random Forest` lebih baik tingkat akurasinya yaitu sebesar **98 %**
