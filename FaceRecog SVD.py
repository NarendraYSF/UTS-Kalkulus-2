import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
import os
from skimage import exposure
from PIL import Image
import matplotlib.cm as cm


class PengenalWajahSVD:
    """
    Kelas untuk implementasi pengenalan wajah menggunakan Singular Value Decomposition (SVD)
    berdasarkan intensitas pencahayaannya.
    """

    def __init__(self, n_components=50):
        """
        Inisialisasi model pengenalan wajah SVD

        Parameter:
        n_components: Jumlah komponen singular yang digunakan (dimensi fitur)
        """
        self.n_components = n_components
        self.mean_face = None
        self.components = None
        self.face_projected = None
        self.labels = None

    def normalisasi_pencahayaan(self, gambar, metode='hist_eq'):
        """
        Melakukan normalisasi pencahayaan pada gambar

        Parameter:
        gambar: Gambar wajah input
        metode: Metode normalisasi ('hist_eq', 'clahe', atau 'gamma')

        Return:
        Gambar yang telah dinormalisasi
        """
        if metode == 'hist_eq':
            # Normalisasi dengan equalisasi histogram
            return exposure.equalize_hist(gambar)
        elif metode == 'clahe':
            # Normalisasi dengan Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if len(gambar.shape) == 2:  # Grayscale
                return clahe.apply(np.uint8(gambar * 255)) / 255.0
            else:  # RGB
                hasil = np.zeros_like(gambar)
                for i in range(gambar.shape[2]):
                    hasil[:, :, i] = clahe.apply(np.uint8(gambar[:, :, i] * 255)) / 255.0
                return hasil
        elif metode == 'gamma':
            # Normalisasi dengan koreksi gamma
            return exposure.adjust_gamma(gambar, gamma=1.2)
        else:
            return gambar

    def latih(self, X, y, metode_normalisasi='hist_eq'):
        """
        Melatih model SVD untuk pengenalan wajah

        Parameter:
        X: Array gambar wajah [n_samples, height, width]
        y: Label identitas untuk setiap gambar
        metode_normalisasi: Metode normalisasi pencahayaan
        """
        n_samples, height, width = X.shape
        X_flat = X.reshape(n_samples, height * width)

        # Normalisasi pencahayaan
        X_norm = np.zeros_like(X_flat)
        for i in range(n_samples):
            img = X_flat[i].reshape(height, width)
            img_norm = self.normalisasi_pencahayaan(img, metode=metode_normalisasi)
            X_norm[i] = img_norm.flatten()

        # Hitung rata-rata wajah
        self.mean_face = np.mean(X_norm, axis=0)

        # Kurangi rata-rata dari setiap gambar
        X_centered = X_norm - self.mean_face

        # Lakukan SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Ambil n_components pertama
        self.components = Vt[:self.n_components]

        # Proyeksikan wajah ke ruang eigen
        self.face_projected = np.dot(X_centered, self.components.T)
        self.labels = y

        # Simpan dimensi gambar untuk rekonstruksi
        self.image_shape = (height, width)

        return self

    def prediksi(self, X, metode_normalisasi='hist_eq', k=1):
        """
        Melakukan prediksi identitas wajah

        Parameter:
        X: Array gambar wajah untuk diprediksi [n_samples, height, width]
        metode_normalisasi: Metode normalisasi pencahayaan
        k: Jumlah tetangga terdekat yang dipertimbangkan

        Return:
        Label identitas hasil prediksi
        """
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        # Normalisasi pencahayaan
        X_norm = np.zeros_like(X_flat)
        for i in range(n_samples):
            img = X_flat[i].reshape(self.image_shape)
            img_norm = self.normalisasi_pencahayaan(img, metode=metode_normalisasi)
            X_norm[i] = img_norm.flatten()

        # Kurangi rata-rata wajah
        X_centered = X_norm - self.mean_face

        # Proyeksikan ke ruang eigen
        X_projected = np.dot(X_centered, self.components.T)

        # Hitung jarak Euclidean ke semua wajah latihan
        prediksi = []
        for i in range(n_samples):
            jarak = np.linalg.norm(self.face_projected - X_projected[i], axis=1)
            idx_terdekat = np.argsort(jarak)[:k]

            # Voting mayoritas untuk k-nearest neighbors
            label_prediksi = np.bincount(self.labels[idx_terdekat]).argmax()
            prediksi.append(label_prediksi)

        return np.array(prediksi)

    def rekonstruksi_wajah(self, X, metode_normalisasi='hist_eq', n_components=None):
        """
        Merekonstruksi wajah dari proyeksi SVD

        Parameter:
        X: Gambar wajah yang akan direkonstruksi
        metode_normalisasi: Metode normalisasi pencahayaan
        n_components: Jumlah komponen yang digunakan

        Return:
        Gambar yang telah direkonstruksi
        """
        if n_components is None:
            n_components = self.n_components

        # Pastikan X dalam bentuk yang tepat
        if len(X.shape) == 3:
            X = X[0]  # Ambil satu gambar saja

        # Flatten dan normalisasi
        X_flat = X.flatten()
        X_norm = self.normalisasi_pencahayaan(X, metode=metode_normalisasi).flatten()

        # Kurangi rata-rata wajah
        X_centered = X_norm - self.mean_face

        # Proyeksikan ke ruang eigen
        X_projected = np.dot(X_centered, self.components[:n_components].T)

        # Rekonstruksi gambar
        X_reconstructed = np.dot(X_projected, self.components[:n_components]) + self.mean_face

        # Reshape kembali ke bentuk gambar
        return X_reconstructed.reshape(self.image_shape)


# Fungsi untuk memuat dataset wajah sederhana
def muat_dataset_wajah(direktori, ukuran=(100, 100)):
    """
    Memuat dataset wajah dari direktori

    Parameter:
    direktori: Path ke direktori dataset
    ukuran: Ukuran untuk resize gambar

    Return:
    X: Array gambar [n_samples, height, width]
    y: Label identitas
    """
    X = []
    y = []
    for label, nama_folder in enumerate(os.listdir(direktori)):
        folder_path = os.path.join(direktori, nama_folder)
        if os.path.isdir(folder_path):
            for nama_file in os.listdir(folder_path):
                if nama_file.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(folder_path, nama_file)
                    try:
                        # Baca gambar dan ubah ke grayscale
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize gambar
                            img = cv2.resize(img, ukuran)
                            # Normalisasi ke [0, 1]
                            img = img / 255.0
                            X.append(img)
                            y.append(label)
                    except Exception as e:
                        print(f"Error membaca {file_path}: {e}")

    return np.array(X), np.array(y)


# Fungsi untuk menampilkan hasil
def tampilkan_hasil(model, X_test, y_test, y_pred, indeks_sampel=None):
    """
    Menampilkan hasil pengenalan wajah

    Parameter:
    model: Model PengenalWajahSVD
    X_test: Data gambar uji
    y_test: Label sebenarnya
    y_pred: Label prediksi
    indeks_sampel: Indeks sampel yang akan ditampilkan
    """
    if indeks_sampel is None:
        # Pilih beberapa sampel acak jika tidak ditentukan
        indeks_sampel = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    # Tampilkan metrik evaluasi
    akurasi = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Akurasi: {akurasi:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Tampilkan contoh hasil
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indeks_sampel):
        # Gambar asli
        plt.subplot(len(indeks_sampel), 3, i * 3 + 1)
        plt.imshow(X_test[idx], cmap='gray')
        plt.title(f"Asli (ID: {y_test[idx]})")
        plt.axis('off')

        # Gambar ternormalisasi
        gambar_norm = model.normalisasi_pencahayaan(X_test[idx])
        plt.subplot(len(indeks_sampel), 3, i * 3 + 2)
        plt.imshow(gambar_norm, cmap='gray')
        plt.title(f"Normalisasi")
        plt.axis('off')

        # Gambar rekonstruksi
        gambar_rekon = model.rekonstruksi_wajah(X_test[idx])
        plt.subplot(len(indeks_sampel), 3, i * 3 + 3)
        plt.imshow(gambar_rekon, cmap='gray')
        plt.title(f"Rekonstruksi (Prediksi: {y_pred[idx]})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Fungsi untuk simulasi pencahayaan berbeda
def simulasi_pencahayaan(gambar, faktor_cahaya=1.0, offset=0.0):
    """
    Melakukan simulasi perubahan intensitas pencahayaan pada gambar

    Parameter:
    gambar: Gambar input
    faktor_cahaya: Faktor pengali intensitas (>1: lebih terang, <1: lebih gelap)
    offset: Nilai offset yang ditambahkan

    Return:
    Gambar dengan intensitas pencahayaan yang diubah
    """
    # Terapkan perubahan intensitas
    gambar_baru = gambar * faktor_cahaya + offset

    # Pastikan nilai tetap dalam range [0, 1]
    gambar_baru = np.clip(gambar_baru, 0, 1)

    return gambar_baru


# Demo simulasi
def demo():
    """
    Fungsi demo untuk menjalankan simulasi pengenalan wajah SVD
    dengan variasi intensitas pencahayaan
    """
    print("Simulasi Pengenalan Wajah dengan SVD Berdasarkan Intensitas Pencahayaan")
    print("=" * 80)

    # Karena kita tidak memiliki dataset yang sebenarnya, kita akan membuat data dummy
    # Dalam implementasi nyata, Anda akan menggunakan dataset wajah yang sebenarnya

    # Buat data dummy (10 orang, 5 gambar per orang)
    n_identitas = 10
    n_sampel_per_identitas = 5
    ukuran_gambar = (100, 100)

    np.random.seed(42)  # Untuk hasil yang konsisten

    # Buat data dummy
    X = np.zeros((n_identitas * n_sampel_per_identitas, ukuran_gambar[0], ukuran_gambar[1]))
    y = np.zeros(n_identitas * n_sampel_per_identitas, dtype=int)

    for i in range(n_identitas):
        # Buat template wajah acak untuk identitas ini
        template_wajah = np.random.rand(ukuran_gambar[0], ukuran_gambar[1]) * 0.5

        # Tambahkan beberapa fitur sederhana (seperti mata, hidung, mulut)
        # Mata
        mata_y = ukuran_gambar[0] // 3
        mata_kiri_x = ukuran_gambar[1] // 3
        mata_kanan_x = 2 * ukuran_gambar[1] // 3
        template_wajah[mata_y - 5:mata_y + 5, mata_kiri_x - 5:mata_kiri_x + 5] = 0.9
        template_wajah[mata_y - 5:mata_y + 5, mata_kanan_x - 5:mata_kanan_x + 5] = 0.9

        # Hidung
        hidung_y = ukuran_gambar[0] // 2
        hidung_x = ukuran_gambar[1] // 2
        template_wajah[hidung_y - 7:hidung_y + 7, hidung_x - 5:hidung_x + 5] = 0.8

        # Mulut
        mulut_y = 2 * ukuran_gambar[0] // 3
        mulut_x = ukuran_gambar[1] // 2
        template_wajah[mulut_y - 3:mulut_y + 3, mulut_x - 15:mulut_x + 15] = 0.85

        # Buat variasi untuk setiap sampel
        for j in range(n_sampel_per_identitas):
            # Indeks dalam array besar
            idx = i * n_sampel_per_identitas + j

            # Tambahkan sedikit noise untuk variasi
            noise = np.random.randn(ukuran_gambar[0], ukuran_gambar[1]) * 0.05

            # Variasikan pencahayaan
            faktor_cahaya = 0.7 + (j / (n_sampel_per_identitas - 1)) * 0.6  # 0.7 hingga 1.3

            # Tambahkan ke dataset
            X[idx] = np.clip(simulasi_pencahayaan(template_wajah, faktor_cahaya) + noise, 0, 1)
            y[idx] = i

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Data training: {X_train.shape}, Data testing: {X_test.shape}")

    # Uji tiga metode normalisasi pencahayaan
    metode_normalisasi = ['hist_eq', 'clahe', 'gamma']
    hasil = {}

    for metode in metode_normalisasi:
        print(f"\nMetode normalisasi: {metode}")

        # Inisialisasi dan latih model
        model = PengenalWajahSVD(n_components=30)
        model.latih(X_train, y_train, metode_normalisasi=metode)

        # Prediksi
        y_pred = model.prediksi(X_test, metode_normalisasi=metode)

        # Evaluasi
        akurasi = accuracy_score(y_test, y_pred)
        hasil[metode] = akurasi

        print(f"Akurasi: {akurasi:.4f}")

        # Tampilkan beberapa sampel
        tampilkan_hasil(model, X_test, y_test, y_pred, indeks_sampel=range(3))

    # Bandingkan metode
    print("\nPerbandingan metode normalisasi:")
    for metode, akurasi in hasil.items():
        print(f"{metode}: {akurasi:.4f}")

    # Uji pengaruh jumlah komponen SVD
    print("\nPengaruh jumlah komponen SVD:")
    metode_terbaik = max(hasil, key=hasil.get)  # Metode dengan akurasi tertinggi
    komponen_list = [5, 10, 20, 30, 40, 50]
    akurasi_komponen = []

    for n_comp in komponen_list:
        model = PengenalWajahSVD(n_components=n_comp)
        model.latih(X_train, y_train, metode_normalisasi=metode_terbaik)
        y_pred = model.prediksi(X_test, metode_normalisasi=metode_terbaik)
        akurasi = accuracy_score(y_test, y_pred)
        akurasi_komponen.append(akurasi)
        print(f"Komponen: {n_comp}, Akurasi: {akurasi:.4f}")

    # Plot pengaruh jumlah komponen
    plt.figure(figsize=(10, 6))
    plt.plot(komponen_list, akurasi_komponen, marker='o')
    plt.xlabel('Jumlah Komponen SVD')
    plt.ylabel('Akurasi')
    plt.title('Pengaruh Jumlah Komponen SVD terhadap Akurasi')
    plt.grid(True)
    plt.show()

    # Visualisasi komponen singular (eigenfaces)
    model_final = PengenalWajahSVD(n_components=10)
    model_final.latih(X_train, y_train, metode_normalisasi=metode_terbaik)

    # Visualisasi eigenfaces
    plt.figure(figsize=(15, 6))
    for i in range(10):  # Tampilkan 10 komponen pertama
        plt.subplot(2, 5, i + 1)
        eigenface = model_final.components[i].reshape(ukuran_gambar)
        plt.imshow(eigenface, cmap='viridis')
        plt.title(f"Komponen #{i + 1}")
        plt.axis('off')

    plt.suptitle('Komponen Singular (Eigenfaces)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()