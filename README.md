# Laporan Proyek Machine Learning - Edwin Mahendra

**Find Me:**
<br>
[Instagram](https://www.instagram.com/edwinmahendra_) or
[GitHub](https://github.com/edwinmahendra) or
[LinkedIn](https://www.linkedin.com/in/edwin-mahendra-a2944821b/)

<hr>

Proyek ini disusun sebagai bagian dari tugas pada *course* Dicoding - Machine Learning Terapan. Dalam proyek ini, sebuah model *Machine Learning* akan dibangun dengan tujuan **mengklasifikasikan lima jenis bunga**, yaitu: **lily, lotus, orchid, sunflower, dan tulip.**

## Domain Proyek

<div align="center"><img src="https://github.com/edwinmahendra/DicodingAssets/blob/main/dataset-cover.jpg?raw=true" width="1000"/></div>
<br>

Proyek ini berfokus pada klasifikasi gambar bunga, suatu topik yang menjadi salah satu penelitian dalam bidang *computer vision* dan *artificial intelligence*. Tujuan utama proyek ini adalah untuk mengembangkan dan menerapkan model *machine learning* yang mampu mengklasifikasikan jenis-jenis bunga dengan akurasi dan efisiensi yang tinggi. 

Dalam hal ini, fokus utama akan diberikan pada **lima jenis bunga** yaitu **lily, lotus, orchid, sunflower, dan tulip**. Masing-masing jenis bunga ini memiliki ciri khas dan unik tersendiri, yang menjadi tantangan tersendiri dalam pengembangan model klasifikasi. Bunga lily adalah bunga yang mempesona dengan enam kelopak yang terbuka lebar dan enam benang sari, biasanya tumbuh dari umbi dengan rentang warna yang luas. Sementara itu, lotus adalah bunga cantik yang tumbuh di atas permukaan air dengan kelopak yang membuka lebar ke atas dan bagian tengah bunga yang menonjol. Orchid, di sisi lain, mencakup beragam jenis dengan simetri bilateral dan lip tengah yang biasanya berbeda warna dan bentuknya dari kelopak lainnya. Bunga Matahari, atau *sunflower*, memikat dengan ukurannya yang besar, lingkaran besar di tengah yang berisi biji-bijian, dan kelopak berwarna kuning cerah yang mengelilinginya. Tulip, dengan bentuknya yang khas, kelopak yang terbentuk sempurna, dan siluet lonceng jika dilihat dari samping, menambah keragaman dalam klasifikasi ini.

Beberapa studi sebelumnya telah melakukan klasifikasi bunga dalam lima kategori: *Daisy, Sunflower, Tulip, Dandelion*, dan *Rose*. Penelitian tersebut menggunakan dataset sebanyak 4323 gambar yang dikumpulkan dari berbagai sumber seperti Kaggle, Google Images, dan Yandex Images. Gambar-gambar ini memiliki variasi ukuran piksel yang berbeda. Metode *machine learning* yang digunakan meliputi Random Forest, KNN, serta beberapa metode *deep learning *seperti VGG, ResNet, dan DenseNet [1]. Studi lainnya, menganalisis model AlexNet dan ResNet dalam klasifikasi citra bunga memanfaatkan transfer learning, menemukan bahwa kedua model tersebut menunjukkan kinerja yang mengesankan dengan tingkat akurasi yang tinggi, meskipun ResNet memperlihatkan keunggulan yang lebih menonjol [2].

## Business Understanding
Proyek ini dirancang untuk menghasilkan analisis model dan model klasifikasi gambar bunga itu sendiri. Berdasarkan luaran tersebut, proyek ini sangat relevan untuk berbagai pihak dengan karakteristik bisnis sebagai berikut:
+ *Developer* aplikasi mobile atau situs web yang menyediakan informasi tentang jenis-jenis bunga dan ingin memudahkan pengguna untuk mencari dan mengidentifikasi bunga melalui gambar.
+ Organisasi lingkungan dan konservasi yang membutuhkan model klasifikasi untuk membantu dalam identifikasi dan katalogisasi spesies bunga.
+ Pengguna umum yang tertarik untuk mengetahui jenis bunga melalui foto yang mereka ambil dengan kamera ponsel mereka.

### Problem Statements

- Bagaimana membangun model *machine learning* yang dapat mencapai akurasi minimal 95% dalam mengklasifikasikan gambar bunga?
- Fitur apa saja yang berperan penting dalam identifikasi jenis bunga dan bagaimana pengaruhnya terhadap akurasi model?
- Apakah algoritma *Deep Learning* seperti *ResNet101V2* lebih efektif dalam klasifikasi gambar bunga dibandingkan dengan model *Convolutional Neural Network (CNN)* konvensional?

### Goals

- Membangun model *machine learning* yang efektif untuk mengklasifikasikan gambar bunga dengan akurasi minimal 95%.
- Mengidentifikasi fitur-fitur penting yang berperan dalam klasifikasi jenis bunga dan memahami pengaruhnya terhadap akurasi model.
- Melakukan analisis dan eksperimen untuk membandingkan efektivitas antara model *Deep Learning* dan model konvensional dalam klasifikasi jenis bunga berdasarkan gambar.

### Solution Statements

Untuk menyiapkan data agar dapat digunakan dalam membangun model klasifikasi bunga, langkah-langkah berikut akan dilakukan:
- **Studi Literatur:** Melakukan analisis terhadap penelitian-penelitian sebelumnya yang berfokus pada klasifikasi bunga, guna memahami metode dan teknik yang telah diterapkan serta potensi tantangan yang mungkin muncul.
- **Eksplorasi Data:** Menyelidiki karakteristik dataset, termasuk jumlah sampel dalam setiap kategori bunga, distribusi fitur-fitur, serta mengidentifikasi kemungkinan adanya *outlier*.
- **Visualisasi Data:** Menerapkan teknik visualisasi data seperti grafik atau plot untuk mempresentasikan distribusi kategori bunga, korelasi antar fitur, dan pola-pola unik yang mungkin ada dalam data.

Dalam menganalisis dan membangun model klasifikasi bunga, solusi yang diusulkan adalah sebagai berikut:
- **Seleksi Model:** Menggunakan dua jenis arsitektur jaringan saraf tiruan, yaitu *base model*/model konvensional CNN dan juga ResNet101V2 yang menjadi *improvement* dari model konvensional
- **Eksperimen dan Evaluasi Model:** Melakukan eksperimen dengan menggunakan kedua model tersebut dan membandingkan performanya dalam hal akurasi, presisi, *recall*, dan *F1-score*. Tujuan penggunaan metrik ini adalah untuk memberikan gambaran yang lebih mendalam dan komprehensif mengenai performa dari kedua model tersebut.

## Data Understanding

Data yang digunakan adalah gambar bunga dari 5 jenis yang berbeda. Setiap kelas memiliki 1000 gambar, sehingga totalnya ada 5000 gambar. Dataset ini dapat diakses secara publik melalui link [Flower Image Dataset by KAUSTHUB KANNAN](https://www.kaggle.com/alxmamaev/flowers-recognition).

Berikut informasi pada dataset:
+ Dataset memiliki format gambar (.jpg)
+ Dataset memiliki ukuran piksel yang bermacam-macam
+ Dataset berisi 5.000 gambar
+ Tiap kelas memiliki jumlah 1.000 gambar sehingga masing-masing kelas memiliki distribusi gambar yang seimbang
+ Dataset memiliki 5 buah kelas
+ Tidak terdapat data gambar yang *corrupt/error*.

### Kelas pada Dataset
Terdapat lima buah kelas pada dataset, yaitu:
+ Lily
+ Lotus
+ Orchid
+ Sunflower
+ Tulip

### Data Visualization

<p align="center">
  <i>Gambar 1. Distribusi gambar</i>
</p>
<div align="center"><img src="https://github.com/edwinmahendra/DicodingAssets/blob/main/visualisasidata_terapan_1.jpeg?raw=true" width="500"/></div>


Pada visualisasi data di atas, ditampilkan distribusi gambar pada setiap kategori atau kelas. Setiap batang vertikal pada diagram batang mewakili jumlah gambar dalam kategori tersebut. Distribusi gambar memberikan gambaran tentang seimbang dengan masing-masing kelas terdapat 1.000 gambar sehingga bila dijumlahkan seluruhnya, maka akan didapatkan hasil sebesar 5.000 data gambar.

### Plot Images
<p align="center">
  <i>Gambar 2. Tampilan 20 gambar yang ada pada dataset</i>
</p>

<div align="center"><img src="https://github.com/edwinmahendra/DicodingAssets/blob/main/plotimages_terapan_1.jpeg?raw=true" width="400"/></div>

Di dalam satu batch gambar dari generator dan menampilkannya menggunakan matplotlib. Ketika dijalankan, baris kode akan menghasilkan **20 gambar** dengan judul yang sesuai dengan nama kelas asli dari label gambar terkait.

## Data Preparation
Persiapan data melibatkan beberapa langkah penting untuk memastikan data siap untuk digunakan dalam model *machine learning*. Proses ini melibatkan pengunduhan dataset, ekstraksi file, pemisahan dataset menjadi *train*, *validation*, dan *test*, serta normalisasi gambar.

### Download Dataset
Dataset dapat diunduh dari sumber yang disediakan. Link untuk mengunduh dataset adalah [disini](http://link-to-dataset.com). Dataset diunduh dengan menggunakan API dari kaggle. 

### Ekstraksi Dataset
Setelah proses pengunduhan dataset selesai, dataset umumnya berbentuk file terkompresi (.zip). Untuk dapat menggunakan data tersebut, proses ekstraksi perlu dilakukan. Ekstraksi ini melibatkan pembukaan file .zip dan pemindahan semua item di dalamnya (dalam hal ini, gambar-gambar bunga) ke direktori tertentu.

### Split Dataset
Setelah proses ekstraksi selesai, dataset yang sekarang terdiri dari gambar bunga perlu dibagi menjadi tiga bagian: data latihan (*train dataset*), data validasi (*validation*), dan data pengujian (*test*). Pembagian ini penting untuk memastikan bahwa model *machine learning* dapat belajar dari sejumlah data (*train dataset*), menyesuaikan proses pembelajaran berdasarkan data lain (*validation dataset*), dan akhirnya diuji kinerjanya pada data yang belum pernah dilihat sebelumnya (*test dataset*).

Pada proyek ini, proses pembagian data dilakukan dengan proporsi 70% untuk data latihan, 20% untuk data validasi, dan 10% untuk data pengujian. Pembagian ini dicapai menggunakan fungsi *splitfolders.ratio* yang memisahkan gambar secara acak ke dalam direktori latihan, validasi, dan pengujian dengan proporsi yang diberikan.

Penting untuk dicatat bahwa pembagian ini dilakukan secara acak untuk memastikan bahwa setiap bagian dari data memiliki distribusi kelas yang serupa. Ini penting untuk mencegah bias dalam model yang mungkin disebabkan oleh distribusi kelas yang tidak seimbang di antara data latihan, validasi, dan pengujian.

### Normalisasi Gambar
Proses ini digunakan untuk mempersiapkan data yang akan digunakan dalam proses pelatihan dan pengujian model machine learning. Dalam tahap ini, dimensi gambar disesuaikan menjadi **224x224 piksel**. Ukuran gambar 224x224 piksel adalah ukuran yang umum digunakan dalam model berbasis CNN. Ukuran ini dipilih karena merupakan *trade-off* antara resolusi gambar dan efisiensi komputasi. Gambar berukuran 224x224 **memberikan cukup informasi** kepada model untuk belajar fitur yang penting, sekaligus **tidak terlalu besar** sehingga membebani memori dan waktu komputasi.

 Selanjutnya, **ukuran batch ditetapkan menjadi 32**. Ini berarti model akan memproses 32 sampel data sekaligus dalam setiap iterasi selama proses pelatihan. Ukuran batch ini dipilih untuk menyeimbangkan **kecepatan konvergensi model** dan penggunaan **memori yang efisien**.

## Modeling

### Konvensional CNN
Pada bagian ini, arsitektur model didefinisikan. Model konvolusional ini dibangun dengan pendekatan *Sequential*, yang memungkinkan penumpukan berbagai layer satu per satu dalam urutan yang telah ditentukan. Model ini terdiri dari berbagai layer yang berbeda, termasuk Conv2D, MaxPooling2D, Dropout, Flatten, dan Dense, masing-masing memiliki peran spesifik dalam proses pembelajaran.

1. **Conv2D (Convolutional 2D Layer)**: Layer ini digunakan untuk mengekstrak fitur dari citra input. Menggunakan fungsi aktivasi **ReLU (Rectified Linear Unit)** yang mengubah semua nilai input negatif menjadi nol, sehingga mempercepat proses konvergensi selama pelatihan.
   
2. **MaxPooling2D**: Layer ini bertujuan untuk mengurangi dimensi spasial (panjang dan lebar) dari volume output sebelumnya, mempertahankan informasi penting saja. Hal ini membantu dalam mengurangi overfitting dan menghitung cost secara lebih efisien.

3. **Dropout**: Teknik ini digunakan untuk mencegah overfitting dalam model. Pada layer ini, sejumlah neuron dipilih secara acak dan dihilangkan selama proses pelatihan. Dropout ini ditetapkan sebesar 0.2 dan 0.5 di beberapa tahap dalam arsitektur model.

4. **Flatten**: Layer ini bertugas untuk mengubah input multidimensi menjadi vektor 1D sebelum memasuki layer Dense.

5. **Dense**: Ini adalah layer neural network yang sepenuhnya terhubung. Layer Dense pertama memiliki 512 neuron dan menggunakan fungsi aktivasi ReLU. Layer Dense kedua, yang juga merupakan layer output, memiliki 5 neuron yang mewakili 5 kelas output dan menggunakan fungsi aktivasi softmax untuk output probabilitas.

Dalam hal kompilasi model, digunakan ***Adam optimizer***. Adam adalah algoritma optimasi yang dapat digunakan sebagai pengganti algoritma *Gradient Descent* stokastik klasik untuk memperbarui bobot iteratif dalam pelatihan jaringan. Untuk fungsi *loss*, digunakan ***'categorical_crossentropy'*** yang cocok untuk masalah klasifikasi *multi-class*. Dan untuk metrik evaluasi, digunakan ***'accuracy'*** untuk menilai seberapa baik model melakukan klasifikasi.

Setelah mendefinisikan model, fungsi ***summary()*** digunakan untuk mendapatkan **gambaran umum tentang struktur model**, termasuk jumlah parameter yang akan dipelajari. Lalu terdapat ***checkpointer*** akan **menyimpan model terbaik** (dengan akurasi validasi tertinggi) selama pelatihan.

Pada model ini, digunakan ***Callback ReduceLROnPlateau*** akan mengurangi *learning rate* saat metrik tertentu berhenti membaik. Tujuannya adalah untuk **menemukan *learning rate* yang optimal** sehingga jika tidak ada peningkatan selama 'patience' epoch, *learning rate* akan dikurangi.

### *Pre-Trained Model: ResNet*

**ResNet101V2** adalah variasi dari model **ResNet** (Residual Network) yang awalnya diperkenalkan oleh Kaiming He, Xiangyu Zhang, Shaoqing Ren, dan Jian Sun dalam makalah mereka "*Deep Residual Learning for Image Recognition*". Arsitektur ini memperkenalkan konsep "*shortcut connections*" atau "*residual connections*", yang memungkinkan aliran gradien lebih efisien selama proses backpropagation dalam pelatihan model. Hal ini membantu dalam mengatasi masalah *vanishing gradient*, yang sering terjadi pada model dengan kedalaman yang sangat tinggi.

**ResNet101V2** adalah versi lebih dalam dari model asli **ResNet**, dengan 101 layer. Versi **V2** dari **ResNet** menghadirkan beberapa peningkatan kecil pada blok residu, yang secara umum memberikan kinerja yang sedikit lebih baik dibandingkan dengan versi asli.

**ResNet101V2** dipilih sebagai model transfer learning dalam proyek ini karena beberapa alasan:

1. **Kedalaman Model**: Dengan 101 layer, model ini cukup dalam untuk mengekstraksi fitur tingkat tinggi dan tingkat rendah dari gambar, yang dapat meningkatkan kinerja pada tugas klasifikasi.
2. **Peningkatan Efisiensi**: Dengan fitur koneksinya, **ResNet101V2** mengatasi masalah hilangnya gradien dan juga memungkinkan peningkatan efisiensi dalam pelatihan dan inferensi.
3. **Pre-trained**: **ResNet101V2** sudah dilatih sebelumnya pada dataset ImageNet, yang berisi lebih dari 1 juta gambar dengan 1000 kategori kelas. Oleh karena itu, model ini sudah memiliki pemahaman yang baik tentang fitur umum dalam gambar.
4. **Kinerja**: **ResNet** secara umum menunjukkan kinerja yang baik pada berbagai tugas klasifikasi gambar, termasuk klasifikasi bunga.

Secara umum, pilihan **ResNet101V2** sebagai model transfer learning berbasis pada kemampuannya mengekstraksi fitur yang berarti dari gambar dan kinerjanya yang terbukti dalam tugas klasifikasi gambar.

Model ResNet101V2 yang digunakan dalam proyek ini menggunakan arsitektur ResNet (*Residual Network*) dengan 101 layer, dengan variasi V2 yang menunjukkan versi kedua dari ResNet. Model ini digunakan dalam bentuk transfer learning, di mana bobot yang telah dilatih sebelumnya pada dataset besar digunakan sebagai titik awal untuk pelatihan model.

- Model ini diawali dengan **ResNet101V2** sebagai model dasar, di mana setiap layer dibuat agar dapat dilatih (*trainable*), kecuali 30 layer terakhir. Faktor penentu dalam pemilihan layer yang dapat dilatih ini adalah bahwa layer awal biasanya mengambil fitur-fitur umum yang dapat digunakan kembali, sedangkan layer terakhir biasanya lebih khusus pada tugas yang spesifik (dalam hal ini, dataset asli yang digunakan untuk melatih ResNet101V2).

- Lapisan **GlobalAveragePooling2D** diterapkan berikutnya, yang memiliki efek mengurangi dimensi spasial dari representasi sebelumnya, dengan merata-ratakan informasi spasial, yang pada akhirnya menghasilkan vektor fitur yang dapat digunakan oleh layer dense.

- Lapisan **Dropout** dengan tingkat dropout 0.5 digunakan sebagai upaya pencegahan *overfitting*. Dropout bekerja dengan mematikan secara acak sejumlah neuron selama proses pelatihan, yang berarti bahwa selama setiap iterasi, sebagian informasi tidak digunakan, yang mendorong model untuk menjadi lebih *robust*.

- Lapisan output, sebuah lapisan **Dense** dengan 5 neuron (mewakili 5 kelas bunga) diterapkan. Fungsi aktivasi yang digunakan adalah softmax, yang menghasilkan vektor probabilitas yang menjumlahkan 1, sangat cocok untuk klasifikasi multikelas. Untuk regularisasi, digunakan L2 regularization, yang mencegah bobot menjadi terlalu besar dan mengakibatkan *overfitting*.

Untuk proses pelatihan:

- **ModelCheckpoint** digunakan untuk menyimpan model dengan akurasi validasi terbaik sepanjang proses pelatihan.
  
- **EarlyStopping** digunakan untuk menghentikan pelatihan jika tidak ada peningkatan pada loss validasi setelah 3 epoch, untuk menghindari *overfitting* dan penggunaan komputasi yang tidak perlu.
  
- Model ini dikompilasi dengan **Adam** sebagai optimizer dengan learning rate 1e-4, fungsi kerugian **categorical_crossentropy** untuk klasifikasi multikelas, dan **accuracy** sebagai metrik.

## Evaluation

### Akurasi Tiap Model Setelah Proses Training
+ **Model Konvensional**<br>
  Tahap evaluasi menggambarkan perubahan akurasi dan loss dari model selama proses pelatihan. **Pada model dasar / konvensional**, akurasi pelatihan mencapai 98% dan akurasi validasi sebesar 83% pada akhir epoch. Namun, peningkatan loss yang signifikan menunjukkan bahwa model belum mencapai titik optimal.
  
  ![Akurasi_CNN](https://github.com/edwinmahendra/DicodingAssets/blob/main/hasilevaluasi_cnn_terapan_1.jpeg?raw=true)

    ```
    Epoch 15: val_accuracy improved from 0.83800 to 0.83900, saving model to saved_models/base_model.hdf5
    110/110 [==============================] - 27s 248ms/step - loss: 0.0752 - accuracy: 0.9760 - val_loss: 0.9635 - val_accuracy: 0.8390 - lr: 2.5000e-04
    ```
+ ***Pre-Trained Model***<br>
  Model transfer learning yang menggunakan arsitektur ResNet menghasilkan performa yang mengesankan. Akurasi mencapai 99% pada data pelatihan dan lebih dari 94% pada data validasi. Sesuai dengan peningkatan akurasi, nilai loss juga relatif kecil, hanya sebesar 18%.

  ![Akurasi_CNN](https://github.com/edwinmahendra/DicodingAssets/blob/main/hasilevaluasi_resnet_terapan_1.jpeg?raw=true)

  ```
  Epoch 14: val_loss did not improve from 0.17982
  32/32 [==============================] - 14s 451ms/step - loss: 0.0398 - accuracy: 0.9910 - val_loss: 0.1876 - val_accuracy: 0.9540
    ```
### Model Accuracy on Test Dataset
Selanjutnya, kinerja kedua model dievaluasi menggunakan data uji. Evaluasi ini menggunakan fungsi evaluate(), yang mengembalikan nilai loss dan metrik yang telah didefinisikan saat model dikompilasi.


**Base Model Loss      :**  0.8013201951980591<br>
**Accuracy Base Model  :**  85.39999723434448

**Model ResNet Loss     :**  0.16583648324012756<br>
**Accuracy ResNet Model :**  95.80000042915344


Hasil evaluasi menunjukkan bahwa model dasar memiliki nilai **loss 0.8013** dan tingkat **akurasi 85.4%**. Di sisi lain, model transfer learning dengan **arsitektur ResNet** mencapai nilai **loss yang lebih rendah, 0.1658**, dan tingkat **akurasi** yang lebih tinggi, **95.8%**. Ini menunjukkan bahwa model transfer learning dengan arsitektur ResNet memiliki kinerja yang lebih baik dibandingkan dengan model dasar.

### Image Prediction Testing
Sebagai langkah berikutnya, prediksi dilakukan terhadap 3 gambar bunga acak dari test dataset menggunakan model ResNet dan hasilnya ditampilkan. Hasil menunjukkan ke-3 gambar yang ada sesuai dengan label aslinya. Berikut adalah hasil pengujian yang dilakukan.

<p align="center">
  <i>Gambar 3. Hasil pengujian gambar</i>
</p>

![Pengujian_Gambar](https://github.com/edwinmahendra/DicodingAssets/blob/main/pengujian3gambar_terapan_1.jpeg?raw=true)
<br>

### Classification Report

+ **Model CNN Konvensional**<br>
  Dari *classification report*, dapat dilihat bahwa model ini memiliki akurasi keseluruhan sebesar 85%. Nilai tertinggi dalam *precision* dan *recall* dicapai oleh kelas *Sunflower* dengan nilai *precision* 0.94 dan *recall* 0.97. Sebaliknya, kelas 'Lilly' dan 'Orchid' memiliki nilai *precision* dan *recall* yang relatif lebih rendah dibandingkan kelas lainnya. Berikut adalah tabel hasil *classification report* untuk model konvensional.

  <p align="center">
  <i>Tabel 1. Hasil classification report pada model CNN Konvensional</i>
  </p>

  <div align="center">

  |               | Precision | Recall | F1-Score | Support |
  | ------------- | --------- | ------ | -------- | ------- |
  | Lilly         | 0.81      | 0.78   | 0.80     | 100     |
  | Lotus         | 0.84      | 0.81   | 0.82     | 100     |
  | Orchid        | 0.80      | 0.88   | 0.84     | 100     |
  | Sunflower     | 0.94      | 0.97   | 0.96     | 100     |
  | Tulip         | 0.88      | 0.83   | 0.86     | 100     |
  | **Accuracy**      |           |        | **0.85**     |         |
  | **Macro Average** | **0.85**      | **0.85**   | **0.85**     |         |
  | **Weighted Avg**  | **0.85**      | **0.85**   | **0.85**     |         |

  </div>

+ ***Pre-Trained Model*** <br>Sementara itu, model ResNet menunjukkan peningkatan kinerja secara signifikan dengan akurasi keseluruhan mencapai 96%. Model ini berhasil mengklasifikasikan kelas 'Sunflower' dengan precision dan recall tertinggi, yaitu 0.99. Seluruh kelas lainnya juga menunjukkan peningkatan kinerja dibandingkan model dasar, dengan nilai precision dan recall yang relatif tinggi.
  
  Secara keseluruhan, berdasarkan laporan klasifikasi, model ResNet menunjukkan kinerja yang lebih baik dalam klasifikasi jenis bunga dibandingkan dengan model dasar. Hal ini ditandai dengan peningkatan akurasi, precision, dan recall pada setiap kelas. Berikut adalah tabel hasil *classification report* untuk *pre-trained model*.

  <p align="center">
  <i>Tabel 2. Hasil classification report pada model Resnet101V2</i>
  </p>

  <div align="center">

  |               | Precision | Recall | F1-Score | Support |
  | ------------- | --------- | ------ | -------- | ------- |
  | Lilly         | 0.95      | 0.88   | 0.91     | 100     |
  | Lotus         | 0.96      | 0.98   | 0.97     | 100     |
  | Orchid        | 0.96      | 0.98   | 0.97     | 100     |
  | Sunflower     | 0.99      | 0.99   | 0.99     | 100     |
  | Tulip         | 0.93      | 0.96   | 0.95     | 100     |
  | **Accuracy**      |           |        | **0.96**     |         |
  | **Macro Average** | **0.96**      | **0.96**   | **0.96**     |         |
  | **Weighted Avg**  | **0.96**      | **0.96**   | **0.96**     |         |

  </div>

+ **Analisis berdasarkan Classification Report**<br>
  Dalam melakukan analisis perbandingan antara *model konvensional* dan *model ResNet*, terdapat **beberapa metrik** yang perlu kita pertimbangkan yaitu **akurasi, presisi, recall, dan F1-score**.

  + **Akurasi**<br> Akurasi adalah rasio prediksi benar (positif dan negatif) dengan keseluruhan data. Dari tabel di atas, dapat kita lihat bahwa model **ResNet memberikan akurasi yang lebih tinggi** yaitu **0.96**, sedangkan model konvensional memiliki akurasi sebesar **0.85**. Oleh karena itu, dalam hal akurasi, model **ResNet lebih unggul**.

  + ***Macro Average* dan *Weighted Average***<br> *Macro Average* menghitung rata-rata metrik untuk setiap kelas tanpa mempertimbangkan proporsi untuk setiap kelas. *Weighted Average* menghitung rata-rata metrik untuk setiap kelas sambil mempertimbangkan jumlah data asli dalam setiap kelas. Pada kedua metrik ini, **model ResNet juga menunjukkan nilai yang lebih tinggi** dibandingkan model konvensional, yaitu sebesar 0.96 dibandingkan 0.85.

  Sedangkan untuk metrik *Precision, Recall,* dan *F1-Score*, kita dapat menganalisisnya secara lebih detail untuk setiap kelas:

  + **Lilly:** Pada kelas ini, model ResNet menunjukkan performa yang lebih baik dibandingkan model konvensional, dengan nilai Precision 0.95, Recall 0.88, dan F1-Score 0.91, dibandingkan dengan model konvensional yang memiliki nilai Precision 0.81, Recall 0.78, dan F1-Score 0.80.

  + **Lotus:** Sama seperti kelas Lilly, **model ResNet juga menunjukkan performa yang lebih baik pada kelas Lotus**, dengan nilai Precision 0.96, Recall 0.98, dan F1-Score 0.97, sedangkan model konvensional memiliki nilai Precision 0.84, Recall 0.81, dan F1-Score 0.82.

  + **Orchid:** Pada kelas ini, **model ResNet masih menunjukkan performa yang lebih baik dibandingkan model konvensional**, dengan nilai Precision 0.96, Recall 0.98, dan F1-Score 0.97, sementara model konvensional memiliki nilai Precision 0.80, Recall 0.88, dan F1-Score 0.84.

  + **Sunflower:** **Model ResNet dan model konvensional menunjukkan performa yang hampir sama** pada kelas ini, namun model ResNet sedikit lebih unggul dengan nilai Precision 0.99, Recall 0.99, dan F1-Score 0.99. Model konvensional menunjukkan nilai Precision 0.94, Recall 0.97, dan F1-Score 0.96.

  + **Tulip:** Pada kelas ini, **model ResNet menunjukkan performa yang lebih baik dibandingkan model konvensional**, dengan nilai Precision 0.93, Recall 0.96, dan F1-Score 0.95, sementara model konvensional memiliki nilai Precision 0.88, Recall 0.83, dan F1-Score 0.86.

Dengan demikian, berdasarkan analisis di atas, dapat disimpulkan bahwa **model ResNet menunjukkan performa yang lebih baik dibandingkan model konvensional** pada setiap kelas dan metrik yang dipertimbangkan.

## Kesimpulan
1. **Penggunaan model *ResNet101V2* menghasilkan peningkatan signifikan dalam kinerja pengklasifikasian gambar bunga dibandingkan dengan model konvensional**. Model *ResNet101V2* menunjukkan kinerja superior dalam metrik evaluasi kunci, termasuk presisi, *recall*, *F1-score*, dan akurasi.

2. *ResNet101V2*, sebagai bagian dari keluarga model *ResNet*, memanfaatkan konsep *residual learning* yang **membantu mengatasi masalah *vanishing gradient***, yang biasanya terjadi pada model *deep learning* dengan banyak layer. Fitur ini memungkinkan model untuk belajar dari gambar secara **lebih efisien dan akurat**.

3. Berdasarkan evaluasi, model *ResNet101V2* berhasil mencapai akurasi sekitar 0.96, yang **jauh lebih baik** dibandingkan dengan model konvensional dengan akurasi sekitar 0.85.

4. Dalam konteks praktis, peningkatan akurasi dan kinerja lainnya ini berarti model *ResNet101V2* dapat digunakan untuk melakukan **pengklasifikasian jenis bunga dengan tingkat akurasi yang tinggi**. Ini bisa sangat berguna dalam berbagai aplikasi, seperti penentuan spesies bunga dalam penelitian botani, pengembangan aplikasi mobile untuk penggemar tanaman, atau bahkan dalam aplikasi komersial, seperti sistem otomatis untuk *sorting* bunga di perusahaan hortikultura.

**Kesimpulan utama** dari proyek ini adalah model *ResNet101V2* menunjukkan kinerja yang sangat baik dalam tugas pengklasifikasian gambar bunga, dan dapat menjadi solusi yang andal dan efektif untuk aplikasi yang membutuhkan pengenalan jenis bunga yang akurat.

## Referensi

[1] B. Chen, J. Liu, J. Liu, and J. Sun, "Flowers Classification via Deep Learning," University of California, San Diego, Department of Electrical and Computer Engineering, 2019.

[2] Falahkhi, B., Achmal, E. F., Rizaldi, M., R.A., R. Rizki, and Yudistira, N., "Perbandingan Model AlexNet dan ResNet dalam Klasifikasi Citra Bunga Memanfaatkan Transfer Learning," in Jurnal Ilmu Komputer Agri-Informatika, vol. , no. 1, pp. 70-78, 2022.
