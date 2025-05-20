import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import csv
import os
from datetime import datetime
import io
from PIL import Image, ImageTk


class AplikasiPrediksiHargaEmas:
    def __init__(self, root):
        self.root = root
        self.root.title("Prediksi Harga Emas - Metode SVD dan Least Square")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Set ikon aplikasi
        try:
            self.root.iconbitmap("gold_icon.ico")
        except:
            pass

        # Data default dari makalah
        self.data_historis = {
            'tahun': [2022, 2023, 2024],
            'inflasi': [3.36, 1.80, 2.09],
            'suku_bunga': [4.0, 5.8125, 6.1042],
            'indeks_usd': [104.0558, 103.4642, 104.46],
            'harga_emas': [1806.9667, 1962.2, 2416.4217]
        }

        # Parameter prediksi (nilai default)
        self.parameter_prediksi = {
            'tahun': 2025,
            'inflasi': 3.5,
            'suku_bunga': 5.5,
            'indeks_usd': 108.0
        }

        # Penyimpanan hasil
        self.hasil = {
            'harga_prediksi': None,
            'komponen_svd': None,
            'vektor_omega': None,
            'perhitungan_manual': 3004.5476876,
            'persentase_galat': None,
            'analisis_sensitivitas': {}
        }

        # Membuat elemen UI utama
        self.buat_ui()

        # Memuat data awal dan memperbarui UI
        self.perbarui_tampilan_data()

    def buat_ui(self):
        """Membuat antarmuka pengguna utama"""
        # Membuat bingkai utama
        self.buat_header()

        # Membuat kontainer utama
        kontainer_utama = ttk.Frame(self.root)
        kontainer_utama.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Menambahkan tab
        self.tab_control = ttk.Notebook(kontainer_utama)

        self.tab_data = ttk.Frame(self.tab_control)
        self.tab_prediksi = ttk.Frame(self.tab_control)
        self.tab_analisis = ttk.Frame(self.tab_control)
        self.tab_teori = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_data, text='Data Historis')
        self.tab_control.add(self.tab_prediksi, text='Prediksi')
        self.tab_control.add(self.tab_analisis, text='Analisis')
        self.tab_control.add(self.tab_teori, text='Teori SVD')

        self.tab_control.pack(expand=1, fill=tk.BOTH)

        # Mengisi setiap tab dengan konten
        self.buat_tab_data()
        self.buat_tab_prediksi()
        self.buat_tab_analisis()
        self.buat_tab_teori()

        # Membuat footer
        self.buat_footer()

    def buat_header(self):
        """Membuat header aplikasi"""
        bingkai_header = ttk.Frame(self.root)
        bingkai_header.pack(fill=tk.X, pady=10)

        label_judul = ttk.Label(
            bingkai_header,
            text="Prediksi Harga Emas menggunakan Metode SVD & Least Square",
            font=("Arial", 16, "bold")
        )
        label_judul.pack()

        label_subjudul = ttk.Label(
            bingkai_header,
            text="Berdasarkan Singular Value Decomposition dan Aljabar Linear",
            font=("Arial", 10)
        )
        label_subjudul.pack()

    def buat_footer(self):
        """Membuat footer aplikasi"""
        bingkai_footer = ttk.Frame(self.root)
        bingkai_footer.pack(fill=tk.X, pady=5)

        label_footer = ttk.Label(
            bingkai_footer,
            text="© 2025 - Berdasarkan penelitian oleh Shannon Aurellius Anastasya Lie - ITB",
            font=("Arial", 8)
        )
        label_footer.pack(side=tk.RIGHT, padx=10)

        # Menambahkan tombol untuk impor/ekspor
        bingkai_tombol = ttk.Frame(bingkai_footer)
        bingkai_tombol.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            bingkai_tombol,
            text="Impor Data",
            command=self.impor_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            bingkai_tombol,
            text="Ekspor Hasil",
            command=self.ekspor_hasil
        ).pack(side=tk.LEFT, padx=5)

    def buat_tab_data(self):
        """Membuat tab data historis"""
        bingkai = ttk.Frame(self.tab_data)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Membagi menjadi dua kolom
        bingkai_kiri = ttk.Frame(bingkai)
        bingkai_kiri.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bingkai_kanan = ttk.Frame(bingkai)
        bingkai_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Tabel data
        ttk.Label(
            bingkai_kiri,
            text="Data Historis (2022-2024)",
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        self.pohon_data = ttk.Treeview(bingkai_kiri, columns=("Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD",
                                                              "Harga Emas (USD)"))
        self.pohon_data.heading("#0", text="")
        self.pohon_data.heading("Tahun", text="Tahun")
        self.pohon_data.heading("Inflasi (%)", text="Inflasi (%)")
        self.pohon_data.heading("Suku Bunga (%)", text="Suku Bunga (%)")
        self.pohon_data.heading("Indeks USD", text="Indeks USD")
        self.pohon_data.heading("Harga Emas (USD)", text="Harga Emas (USD)")

        self.pohon_data.column("#0", width=0, stretch=tk.NO)
        self.pohon_data.column("Tahun", width=80, anchor=tk.CENTER)
        self.pohon_data.column("Inflasi (%)", width=100, anchor=tk.CENTER)
        self.pohon_data.column("Suku Bunga (%)", width=100, anchor=tk.CENTER)
        self.pohon_data.column("Indeks USD", width=100, anchor=tk.CENTER)
        self.pohon_data.column("Harga Emas (USD)", width=120, anchor=tk.CENTER)

        self.pohon_data.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tambahkan tombol untuk pengeditan data
        bingkai_edit = ttk.Frame(bingkai_kiri)
        bingkai_edit.pack(fill=tk.X, pady=5)

        ttk.Button(
            bingkai_edit,
            text="Tambah Baris",
            command=self.tambah_baris_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            bingkai_edit,
            text="Edit Data Terpilih",
            command=self.edit_baris_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            bingkai_edit,
            text="Hapus Data Terpilih",
            command=self.hapus_baris_data
        ).pack(side=tk.LEFT, padx=5)

        # Visualisasi
        ttk.Label(
            bingkai_kanan,
            text="Visualisasi Data",
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        # Membuat gambar matplotlib untuk visualisasi data
        self.gambar_data = Figure(figsize=(6, 8), dpi=100)
        self.kanvas_data = FigureCanvasTkAgg(self.gambar_data, bingkai_kanan)
        self.kanvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tambahkan kontrol visualisasi
        kontrol_viz = ttk.Frame(bingkai_kanan)
        kontrol_viz.pack(fill=tk.X, pady=5)

        ttk.Label(kontrol_viz, text="Jenis Grafik:").pack(side=tk.LEFT, padx=5)

        self.jenis_plot = tk.StringVar(value="line")
        combo_plot = ttk.Combobox(
            kontrol_viz,
            textvariable=self.jenis_plot,
            values=["line", "bar", "scatter"],
            width=10,
            state="readonly"
        )
        combo_plot.pack(side=tk.LEFT, padx=5)
        combo_plot.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_data)

        ttk.Button(
            kontrol_viz,
            text="Perbarui Grafik",
            command=self.perbarui_visualisasi_data
        ).pack(side=tk.LEFT, padx=5)

    def buat_tab_prediksi(self):
        """Membuat tab prediksi"""
        bingkai = ttk.Frame(self.tab_prediksi)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Membagi menjadi dua kolom
        bingkai_kiri = ttk.Frame(bingkai)
        bingkai_kiri.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bingkai_kanan = ttk.Frame(bingkai)
        bingkai_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Parameter prediksi
        bingkai_param = ttk.LabelFrame(bingkai_kiri, text="Parameter Prediksi")
        bingkai_param.pack(fill=tk.X, pady=10, padx=5)

        # Input tahun
        ttk.Label(bingkai_param, text="Tahun:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_tahun = tk.StringVar(value=str(self.parameter_prediksi['tahun']))
        ttk.Entry(bingkai_param, textvariable=self.var_tahun, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Input inflasi
        ttk.Label(bingkai_param, text="Inflasi (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_inflasi = tk.StringVar(value=str(self.parameter_prediksi['inflasi']))
        ttk.Entry(bingkai_param, textvariable=self.var_inflasi, width=10).grid(row=1, column=1, padx=5, pady=5)

        # Input suku bunga
        ttk.Label(bingkai_param, text="Suku Bunga (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_suku_bunga = tk.StringVar(value=str(self.parameter_prediksi['suku_bunga']))
        ttk.Entry(bingkai_param, textvariable=self.var_suku_bunga, width=10).grid(row=2, column=1, padx=5, pady=5)

        # Input indeks USD
        ttk.Label(bingkai_param, text="Indeks USD:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_usd = tk.StringVar(value=str(self.parameter_prediksi['indeks_usd']))
        ttk.Entry(bingkai_param, textvariable=self.var_usd, width=10).grid(row=3, column=1, padx=5, pady=5)

        # Tombol prediksi
        ttk.Button(
            bingkai_kiri,
            text="Hitung Prediksi",
            command=self.jalankan_prediksi
        ).pack(pady=10)

        # Tampilan komponen SVD
        bingkai_svd = ttk.LabelFrame(bingkai_kiri, text="Komponen SVD")
        bingkai_svd.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        self.teks_svd = tk.Text(bingkai_svd, height=15, width=40)
        self.teks_svd.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tampilan hasil
        bingkai_hasil = ttk.LabelFrame(bingkai_kanan, text="Hasil Prediksi")
        bingkai_hasil.pack(fill=tk.X, pady=10, padx=5)

        # Tampilan hasil besar
        self.label_hasil = ttk.Label(
            bingkai_hasil,
            text="Jalankan prediksi untuk melihat hasil",
            font=("Arial", 20, "bold")
        )
        self.label_hasil.pack(pady=20)

        # Hasil tambahan
        self.teks_detail = tk.Text(bingkai_hasil, height=10, width=40)
        self.teks_detail.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Grafik prediksi
        bingkai_grafik = ttk.LabelFrame(bingkai_kanan, text="Proyeksi Harga")
        bingkai_grafik.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        self.gambar_pred = Figure(figsize=(5, 4), dpi=100)
        self.kanvas_pred = FigureCanvasTkAgg(self.gambar_pred, bingkai_grafik)
        self.kanvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def buat_tab_analisis(self):
        """Membuat tab analisis"""
        bingkai = ttk.Frame(self.tab_analisis)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Membagi menjadi dua kolom
        bingkai_kiri = ttk.Frame(bingkai)
        bingkai_kiri.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bingkai_kanan = ttk.Frame(bingkai)
        bingkai_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Analisis sensitivitas
        bingkai_sensitivitas = ttk.LabelFrame(bingkai_kiri, text="Analisis Sensitivitas")
        bingkai_sensitivitas.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        bingkai_kontrol = ttk.Frame(bingkai_sensitivitas)
        bingkai_kontrol.pack(fill=tk.X, pady=5)

        ttk.Label(bingkai_kontrol, text="Parameter:").pack(side=tk.LEFT, padx=5)

        self.param_sensitivitas = tk.StringVar(value="inflasi")
        combo_param = ttk.Combobox(
            bingkai_kontrol,
            textvariable=self.param_sensitivitas,
            values=["inflasi", "suku_bunga", "indeks_usd"],
            width=15,
            state="readonly"
        )
        combo_param.pack(side=tk.LEFT, padx=5)

        ttk.Label(bingkai_kontrol, text="Rentang (±%):").pack(side=tk.LEFT, padx=5)

        self.rentang_sensitivitas = tk.StringVar(value="20")
        ttk.Entry(
            bingkai_kontrol,
            textvariable=self.rentang_sensitivitas,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(bingkai_kontrol, text="Langkah:").pack(side=tk.LEFT, padx=5)

        self.langkah_sensitivitas = tk.StringVar(value="10")
        ttk.Entry(
            bingkai_kontrol,
            textvariable=self.langkah_sensitivitas,
            width=5
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            bingkai_kontrol,
            text="Jalankan Analisis",
            command=self.jalankan_analisis_sensitivitas
        ).pack(side=tk.LEFT, padx=10)

        # Grafik sensitivitas
        self.gambar_sensitivitas = Figure(figsize=(5, 4), dpi=100)
        self.kanvas_sensitivitas = FigureCanvasTkAgg(self.gambar_sensitivitas, bingkai_sensitivitas)
        self.kanvas_sensitivitas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Analisis korelasi
        bingkai_korelasi = ttk.LabelFrame(bingkai_kanan, text="Analisis Korelasi")
        bingkai_korelasi.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

        self.gambar_korelasi = Figure(figsize=(5, 4), dpi=100)
        self.kanvas_korelasi = FigureCanvasTkAgg(self.gambar_korelasi, bingkai_korelasi)
        self.kanvas_korelasi.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Analisis galat
        bingkai_galat = ttk.LabelFrame(bingkai_kanan, text="Analisis Galat")
        bingkai_galat.pack(fill=tk.X, pady=10, padx=5)

        self.teks_galat = tk.Text(bingkai_galat, height=6, width=40)
        self.teks_galat.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def buat_tab_teori(self):
        """Membuat tab teori SVD"""
        bingkai = ttk.Frame(self.tab_teori)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Membuat notebook untuk berbagai bagian teori
        notebook_teori = ttk.Notebook(bingkai)
        notebook_teori.pack(fill=tk.BOTH, expand=True)

        # Tab Ikhtisar SVD
        tab_ikhtisar = ttk.Frame(notebook_teori)
        notebook_teori.add(tab_ikhtisar, text="Ikhtisar SVD")

        teks_ikhtisar = """
        # Ikhtisar Singular Value Decomposition (SVD)

        SVD adalah faktorisasi matriks yang menggeneralisasi eigendecomposition dari matriks persegi ke matriks m×n apa pun. Ini adalah cara untuk mendekomposisi matriks menjadi tiga matriks lain:

        M = U∑V^T

        Dimana:
        - M adalah matriks asli (m×n)
        - U adalah matriks vektor singular kiri (m×m)
        - ∑ adalah matriks diagonal yang berisi nilai singular (m×n)
        - V^T adalah transpose dari matriks vektor singular kanan (n×n)

        SVD memiliki banyak aplikasi praktis, termasuk:
        - Reduksi dimensi
        - Pengurangan noise
        - Kompresi data
        - Menyelesaikan masalah least squares
        - Sistem rekomendasi

        Dalam aplikasi prediksi harga emas kita, SVD membantu kami mengidentifikasi pola dasar dalam data historis, mengurangi dimensi sambil mempertahankan informasi penting yang diperlukan untuk prediksi.
        """

        scroll_ikhtisar = ttk.Scrollbar(tab_ikhtisar)
        scroll_ikhtisar.pack(side=tk.RIGHT, fill=tk.Y)

        tampilan_ikhtisar = tk.Text(tab_ikhtisar, yscrollcommand=scroll_ikhtisar.set, wrap=tk.WORD)
        tampilan_ikhtisar.pack(fill=tk.BOTH, expand=True)
        tampilan_ikhtisar.insert(tk.END, teks_ikhtisar)
        tampilan_ikhtisar.config(state=tk.DISABLED)

        scroll_ikhtisar.config(command=tampilan_ikhtisar.yview)

        # Tab Matematika
        tab_matematika = ttk.Frame(notebook_teori)
        notebook_teori.add(tab_matematika, text="Matematika")

        teks_matematika = """
        # Detail Matematika SVD

        ## Dekomposisi SVD
        Untuk matriks M dengan ukuran m×n, dekomposisi SVD diberikan oleh:

        M = U∑V^T

        Dimana:
        - U (m×m) adalah matriks ortogonal yang kolomnya adalah eigenvector dari MM^T
        - ∑ (m×n) adalah matriks diagonal dengan entri non-nol berupa akar kuadrat dari eigenvalue dari M^TM atau MM^T
        - V^T (n×n) adalah transpose dari matriks ortogonal V yang kolomnya adalah eigenvector dari M^TM

        ## Menghitung SVD Secara Manual
        1. Hitung M^TM dan temukan eigenvalue dan eigenvector-nya
        2. Eigenvector dari M^TM adalah kolom-kolom dari V
        3. Nilai singular dalam ∑ adalah akar kuadrat dari eigenvalue dari M^TM
        4. Hitung U = MV∑^(-1)

        ## Metode Least Squares dengan SVD
        Solusi least squares untuk sistem Mω = y dapat dihitung menggunakan SVD sebagai:

        ω = V∑^(+)U^Ty

        Dimana ∑^(+) adalah pseudoinverse dari ∑, diperoleh dengan mengambil kebalikan dari setiap elemen diagonal non-nol, membiarkan nol di tempatnya, dan mentranspose matriks yang dihasilkan.

        ## Rumus Prediksi
        Prediksi untuk harga emas di masa depan dihitung sebagai:

        y_prediksi = M_masa_depan * ω

        Dimana M_masa_depan berisi indikator ekonomi untuk tahun yang ingin kita prediksi.
        """

        scroll_matematika = ttk.Scrollbar(tab_matematika)
        scroll_matematika.pack(side=tk.RIGHT, fill=tk.Y)

        tampilan_matematika = tk.Text(tab_matematika, yscrollcommand=scroll_matematika.set, wrap=tk.WORD)
        tampilan_matematika.pack(fill=tk.BOTH, expand=True)
        tampilan_matematika.insert(tk.END, teks_matematika)
        tampilan_matematika.config(state=tk.DISABLED)

        scroll_matematika.config(command=tampilan_matematika.yview)

        # Tab Implementasi
        tab_impl = ttk.Frame(notebook_teori)
        notebook_teori.add(tab_impl, text="Implementasi")

        teks_impl = """
        # Implementasi SVD dalam Python

        Kode berikut menunjukkan cara mengimplementasikan SVD untuk prediksi harga emas:

        ```python
        def prediksi_svd_least_square(M, y, Mx):
            # Langkah 1: Hitung SVD dari matriks M
            U, S, VT = np.linalg.svd(M, full_matrices=False)
            V = VT.T

            # Langkah 2: Hitung U^T * y
            UT_y = U.T @ y

            # Langkah 3: Hitung Sigma^+ (pseudo-inverse) * (U^T * y)
            S_inv = np.zeros_like(S)
            for i in range(len(S)):
                if S[i] > 0:
                    S_inv[i] = 1 / S[i]

            Sigma_plus_UT_y = S_inv * UT_y  # Perkalian elemen per elemen

            # Langkah 4: Hitung omega = V * Sigma^+ * (U^T * y)
            omega = V @ Sigma_plus_UT_y

            # Langkah 5: Hitung prediksi menggunakan Mx * omega
            y_pred = Mx @ omega

            return y_pred
        ```

        ## Contoh Numerik
        Berdasarkan data historis dari 2022-2024:

        M = [
            [3.36, 4, 104.0558],
            [1.80, 5.8125, 103.4642],
            [2.09, 6.1042, 104.46]
        ]

        y = [1806.9667, 1962.2, 2416.4217]

        Dan parameter masa depan untuk 2025:

        M_2025 = [3.5, 5.50, 108]

        Proses prediksi mengikuti langkah-langkah berikut:
        1. Dekomposisi M menggunakan SVD untuk mendapatkan U, ∑, dan V
        2. Hitung U^T * y
        3. Kalikan dengan pseudoinverse dari ∑
        4. Kalikan dengan V untuk mendapatkan ω
        5. Kalikan M_2025 dengan ω untuk mendapatkan harga emas yang diprediksi

        Hasilnya memberi kita prediksi harga emas untuk tahun 2025.
        """

        scroll_impl = ttk.Scrollbar(tab_impl)
        scroll_impl.pack(side=tk.RIGHT, fill=tk.Y)

        tampilan_impl = tk.Text(tab_impl, yscrollcommand=scroll_impl.set, wrap=tk.WORD)
        tampilan_impl.pack(fill=tk.BOTH, expand=True)
        tampilan_impl.insert(tk.END, teks_impl)
        tampilan_impl.config(state=tk.DISABLED)

        scroll_impl.config(command=tampilan_impl.yview)

        # Tab Referensi
        tab_ref = ttk.Frame(notebook_teori)
        notebook_teori.add(tab_ref, text="Referensi")

        teks_ref = """
        # Referensi

        1. Shannon Aurellius Anastasya Lie (2025). "Penerapan Singular Value Decomposition (SVD) dan Metode Least Square dalam Prediksi Harga Emas 2025". Institut Teknologi Bandung.

        2. Golub, G. H., & Reinsch, C. (1970). "Singular value decomposition and least squares solutions". Numerische mathematik, 14(5), 403-420.

        3. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). "Numerical recipes: The art of scientific computing" (3rd ed.). Cambridge university press.

        4. Strang, G. (2016). "Introduction to Linear Algebra" (5th ed.). Wellesley-Cambridge Press.

        5. Rinaldi Munir, "Singular Value Decomposition (SVD)". Sekolah Teknik Elektro dan Informatika (STEI) ITB.

        6. Bank Indonesia, "Data Inflasi." Statistik dan Indikator Ekonomi.

        7. Investing.com, "Gold Historical Data" dan "USDollar Historical Data".
        """

        scroll_ref = ttk.Scrollbar(tab_ref)
        scroll_ref.pack(side=tk.RIGHT, fill=tk.Y)

        tampilan_ref = tk.Text(tab_ref, yscrollcommand=scroll_ref.set, wrap=tk.WORD)
        tampilan_ref.pack(fill=tk.BOTH, expand=True)
        tampilan_ref.insert(tk.END, teks_ref)
        tampilan_ref.config(state=tk.DISABLED)

        scroll_ref.config(command=tampilan_ref.yview)

    def perbarui_tampilan_data(self):
        """Memperbarui tampilan data di UI"""
        # Membersihkan treeview
        for item in self.pohon_data.get_children():
            self.pohon_data.delete(item)

        # Menambahkan data ke treeview
        for i in range(len(self.data_historis['tahun'])):
            self.pohon_data.insert(
                "",
                "end",
                values=(
                    self.data_historis['tahun'][i],
                    self.data_historis['inflasi'][i],
                    self.data_historis['suku_bunga'][i],
                    self.data_historis['indeks_usd'][i],
                    self.data_historis['harga_emas'][i]
                )
            )

        # Memperbarui visualisasi
        self.perbarui_visualisasi_data()

        # Memperbarui analisis korelasi
        self.perbarui_analisis_korelasi()

    def perbarui_visualisasi_data(self, event=None):
        """Memperbarui visualisasi data berdasarkan jenis plot yang dipilih"""
        jenis_plot = self.jenis_plot.get()

        # Menghapus plot sebelumnya
        self.gambar_data.clear()

        # Membuat subplot
        axes = []
        axes.append(self.gambar_data.add_subplot(411))  # Harga emas
        axes.append(self.gambar_data.add_subplot(412))  # Inflasi
        axes.append(self.gambar_data.add_subplot(413))  # Suku bunga
        axes.append(self.gambar_data.add_subplot(414))  # Indeks USD

        tahun = self.data_historis['tahun']

        # Plot harga emas
        if jenis_plot == "line":
            axes[0].plot(tahun, self.data_historis['harga_emas'], 'o-', color='gold', linewidth=2)
        elif jenis_plot == "bar":
            axes[0].bar(tahun, self.data_historis['harga_emas'], color='gold', alpha=0.7)
        else:  # scatter
            axes[0].scatter(tahun, self.data_historis['harga_emas'], color='gold', s=100)

        axes[0].set_title('Harga Emas (USD)')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Plot inflasi
        if jenis_plot == "line":
            axes[1].plot(tahun, self.data_historis['inflasi'], 'o-', color='red', linewidth=2)
        elif jenis_plot == "bar":
            axes[1].bar(tahun, self.data_historis['inflasi'], color='red', alpha=0.7)
        else:  # scatter
            axes[1].scatter(tahun, self.data_historis['inflasi'], color='red', s=100)

        axes[1].set_title('Inflasi (%)')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Plot suku bunga
        if jenis_plot == "line":
            axes[2].plot(tahun, self.data_historis['suku_bunga'], 'o-', color='blue', linewidth=2)
        elif jenis_plot == "bar":
            axes[2].bar(tahun, self.data_historis['suku_bunga'], color='blue', alpha=0.7)
        else:  # scatter
            axes[2].scatter(tahun, self.data_historis['suku_bunga'], color='blue', s=100)

        axes[2].set_title('Suku Bunga (%)')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Plot indeks USD
        if jenis_plot == "line":
            axes[3].plot(tahun, self.data_historis['indeks_usd'], 'o-', color='green', linewidth=2)
        elif jenis_plot == "bar":
            axes[3].bar(tahun, self.data_historis['indeks_usd'], color='green', alpha=0.7)
        else:  # scatter
            axes[3].scatter(tahun, self.data_historis['indeks_usd'], color='green', s=100)

        axes[3].set_title('Indeks USD')
        axes[3].grid(True, linestyle='--', alpha=0.7)

        self.gambar_data.tight_layout()
        self.kanvas_data.draw()

    def perbarui_visualisasi_prediksi(self):
        """Memperbarui visualisasi prediksi"""
        if self.hasil['harga_prediksi'] is None:
            return

        # Menghapus plot sebelumnya
        self.gambar_pred.clear()

        # Membuat plot
        ax = self.gambar_pred.add_subplot(111)

        # Membuat sumbu x dengan tahun
        tahun = self.data_historis['tahun'] + [self.parameter_prediksi['tahun']]
        harga = self.data_historis['harga_emas'] + [self.hasil['harga_prediksi']]

        # Plot data historis
        ax.plot(
            self.data_historis['tahun'],
            self.data_historis['harga_emas'],
            'o-',
            color='blue',
            linewidth=2,
            label='Historis'
        )

        # Plot prediksi
        ax.plot(
            [self.data_historis['tahun'][-1], self.parameter_prediksi['tahun']],
            [self.data_historis['harga_emas'][-1], self.hasil['harga_prediksi']],
            'o--',
            color='red',
            linewidth=2,
            label='Prediksi'
        )

        # Sorot titik prediksi
        ax.scatter(
            [self.parameter_prediksi['tahun']],
            [self.hasil['harga_prediksi']],
            color='red',
            s=100,
            zorder=5
        )

        # Tambahkan anotasi nilai
        ax.annotate(
            f"${self.hasil['harga_prediksi']:.2f}",
            xy=(self.parameter_prediksi['tahun'], self.hasil['harga_prediksi']),
            xytext=(10, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
        )

        ax.set_title(f'Prediksi Harga Emas untuk tahun {self.parameter_prediksi["tahun"]}')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Harga Emas (USD)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Atur sumbu x untuk hanya menampilkan tahun
        ax.set_xticks(tahun)
        ax.set_xticklabels([str(tahun) for tahun in tahun])

        self.gambar_pred.tight_layout()
        self.kanvas_pred.draw()

    def perbarui_analisis_korelasi(self):
        """Memperbarui visualisasi analisis korelasi"""
        # Menghapus plot sebelumnya
        self.gambar_korelasi.clear()

        # Membuat matriks korelasi
        data = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd'],
            self.data_historis['harga_emas']
        ])

        label = ['Inflasi', 'Suku Bunga', 'Indeks USD', 'Harga Emas']

        # Menghitung matriks korelasi
        matriks_korelasi = np.corrcoef(data)

        # Membuat heatmap
        ax = self.gambar_korelasi.add_subplot(111)
        im = ax.imshow(matriks_korelasi, cmap='coolwarm', vmin=-1, vmax=1)

        # Tambahkan colorbar
        cbar = self.gambar_korelasi.colorbar(im)
        cbar.set_label('Koefisien Korelasi')

        # Tambahkan label
        ax.set_xticks(np.arange(len(label)))
        ax.set_yticks(np.arange(len(label)))
        ax.set_xticklabels(label)
        ax.set_yticklabels(label)

        # Putar label sumbu x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Tambahkan nilai korelasi dalam sel
        for i in range(len(label)):
            for j in range(len(label)):
                text = ax.text(j, i, f"{matriks_korelasi[i, j]:.2f}",
                               ha="center", va="center",
                               color="black" if abs(matriks_korelasi[i, j]) < 0.7 else "white")

        ax.set_title("Matriks Korelasi")

        self.gambar_korelasi.tight_layout()
        self.kanvas_korelasi.draw()

    def perbarui_analisis_sensitivitas(self):
        """Memperbarui visualisasi analisis sensitivitas berdasarkan hasil"""
        if not self.hasil['analisis_sensitivitas']:
            return

        # Menghapus plot sebelumnya
        self.gambar_sensitivitas.clear()

        # Membuat plot
        ax = self.gambar_sensitivitas.add_subplot(111)

        param = self.param_sensitivitas.get()
        nilai = self.hasil['analisis_sensitivitas']['nilai']
        harga = self.hasil['analisis_sensitivitas']['harga']
        dasar = self.hasil['harga_prediksi']

        # Plot kurva sensitivitas
        ax.plot(nilai, harga, 'o-', color='purple', linewidth=2)

        # Sorot baseline
        baseline_idx = len(nilai) // 2
        ax.scatter([nilai[baseline_idx]], [harga[baseline_idx]], color='red', s=100, zorder=5)

        # Tambahkan anotasi nilai
        ax.annotate(
            f"Dasar: ${dasar:.2f}",
            xy=(nilai[baseline_idx], harga[baseline_idx]),
            xytext=(10, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
        )

        # Atur judul dan label
        label_param = {
            'inflasi': 'Inflasi (%)',
            'suku_bunga': 'Suku Bunga (%)',
            'indeks_usd': 'Indeks USD'
        }

        ax.set_title(f'Analisis Sensitivitas: {label_param[param]}')
        ax.set_xlabel(label_param[param])
        ax.set_ylabel('Harga Emas Prediksi (USD)')
        ax.grid(True, linestyle='--', alpha=0.7)

        self.gambar_sensitivitas.tight_layout()
        self.kanvas_sensitivitas.draw()

    def tambah_baris_data(self):
        """Menambahkan baris data baru ke data historis"""
        # Membuat jendela popup untuk entri data
        popup = tk.Toplevel()
        popup.title("Tambah Baris Data")
        popup.geometry("300x250")
        popup.grab_set()  # Membuat jendela modal

        # Field entri
        ttk.Label(popup, text="Tahun:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        var_tahun = tk.StringVar()
        ttk.Entry(popup, textvariable=var_tahun).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Inflasi (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        var_inflasi = tk.StringVar()
        ttk.Entry(popup, textvariable=var_inflasi).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Suku Bunga (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        var_suku_bunga = tk.StringVar()
        ttk.Entry(popup, textvariable=var_suku_bunga).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Indeks USD:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        var_usd = tk.StringVar()
        ttk.Entry(popup, textvariable=var_usd).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Harga Emas (USD):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        var_emas = tk.StringVar()
        ttk.Entry(popup, textvariable=var_emas).grid(row=4, column=1, padx=5, pady=5)

        # Fungsi untuk menyimpan data
        def simpan_data():
            try:
                tahun = int(var_tahun.get())
                inflasi = float(var_inflasi.get())
                suku_bunga = float(var_suku_bunga.get())
                usd = float(var_usd.get())
                emas = float(var_emas.get())

                # Menambahkan data ke penyimpanan kita
                self.data_historis['tahun'].append(tahun)
                self.data_historis['inflasi'].append(inflasi)
                self.data_historis['suku_bunga'].append(suku_bunga)
                self.data_historis['indeks_usd'].append(usd)
                self.data_historis['harga_emas'].append(emas)

                # Mengurutkan data berdasarkan tahun
                indeks = np.argsort(self.data_historis['tahun'])
                self.data_historis['tahun'] = [self.data_historis['tahun'][i] for i in indeks]
                self.data_historis['inflasi'] = [self.data_historis['inflasi'][i] for i in indeks]
                self.data_historis['suku_bunga'] = [self.data_historis['suku_bunga'][i] for i in indeks]
                self.data_historis['indeks_usd'] = [self.data_historis['indeks_usd'][i] for i in indeks]
                self.data_historis['harga_emas'] = [self.data_historis['harga_emas'][i] for i in indeks]

                # Memperbarui tampilan
                self.perbarui_tampilan_data()

                # Menutup popup
                popup.destroy()

            except ValueError:
                messagebox.showerror("Kesalahan Input", "Mohon masukkan nilai numerik yang valid")

        # Menambahkan tombol
        ttk.Button(popup, text="Simpan", command=simpan_data).grid(row=5, column=0, padx=5, pady=20)
        ttk.Button(popup, text="Batal", command=popup.destroy).grid(row=5, column=1, padx=5, pady=20)

    def edit_baris_data(self):
        """Mengedit baris data yang dipilih"""
        selected = self.pohon_data.selection()
        if not selected:
            messagebox.showinfo("Pilihan", "Mohon pilih baris untuk diedit")
            return

        # Mendapatkan indeks item yang dipilih
        item = self.pohon_data.item(selected[0])
        nilai = item['values']
        indeks = self.data_historis['tahun'].index(nilai[0])

        # Membuat jendela popup untuk mengedit data
        popup = tk.Toplevel()
        popup.title("Edit Baris Data")
        popup.geometry("300x250")
        popup.grab_set()  # Membuat jendela modal

        # Field entri
        ttk.Label(popup, text="Tahun:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        var_tahun = tk.StringVar(value=str(nilai[0]))
        ttk.Entry(popup, textvariable=var_tahun).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Inflasi (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        var_inflasi = tk.StringVar(value=str(nilai[1]))
        ttk.Entry(popup, textvariable=var_inflasi).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Suku Bunga (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        var_suku_bunga = tk.StringVar(value=str(nilai[2]))
        ttk.Entry(popup, textvariable=var_suku_bunga).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Indeks USD:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        var_usd = tk.StringVar(value=str(nilai[3]))
        ttk.Entry(popup, textvariable=var_usd).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(popup, text="Harga Emas (USD):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        var_emas = tk.StringVar(value=str(nilai[4]))
        ttk.Entry(popup, textvariable=var_emas).grid(row=4, column=1, padx=5, pady=5)

        # Fungsi untuk menyimpan data
        def simpan_data():
            try:
                tahun = int(var_tahun.get())
                inflasi = float(var_inflasi.get())
                suku_bunga = float(var_suku_bunga.get())
                usd = float(var_usd.get())
                emas = float(var_emas.get())

                # Memperbarui data di penyimpanan kita
                self.data_historis['tahun'][indeks] = tahun
                self.data_historis['inflasi'][indeks] = inflasi
                self.data_historis['suku_bunga'][indeks] = suku_bunga
                self.data_historis['indeks_usd'][indeks] = usd
                self.data_historis['harga_emas'][indeks] = emas

                # Mengurutkan data berdasarkan tahun
                indeks_urut = np.argsort(self.data_historis['tahun'])
                self.data_historis['tahun'] = [self.data_historis['tahun'][i] for i in indeks_urut]
                self.data_historis['inflasi'] = [self.data_historis['inflasi'][i] for i in indeks_urut]
                self.data_historis['suku_bunga'] = [self.data_historis['suku_bunga'][i] for i in indeks_urut]
                self.data_historis['indeks_usd'] = [self.data_historis['indeks_usd'][i] for i in indeks_urut]
                self.data_historis['harga_emas'] = [self.data_historis['harga_emas'][i] for i in indeks_urut]

                # Memperbarui tampilan
                self.perbarui_tampilan_data()

                # Menutup popup
                popup.destroy()

            except ValueError:
                messagebox.showerror("Kesalahan Input", "Mohon masukkan nilai numerik yang valid")

        # Menambahkan tombol
        ttk.Button(popup, text="Simpan", command=simpan_data).grid(row=5, column=0, padx=5, pady=20)
        ttk.Button(popup, text="Batal", command=popup.destroy).grid(row=5, column=1, padx=5, pady=20)

    def hapus_baris_data(self):
        """Menghapus baris data yang dipilih"""
        selected = self.pohon_data.selection()
        if not selected:
            messagebox.showinfo("Pilihan", "Mohon pilih baris untuk dihapus")
            return

        # Konfirmasi penghapusan
        if not messagebox.askyesno("Konfirmasi", "Apakah Anda yakin ingin menghapus baris ini?"):
            return

        # Mendapatkan indeks item yang dipilih
        item = self.pohon_data.item(selected[0])
        nilai = item['values']
        indeks = self.data_historis['tahun'].index(nilai[0])

        # Menghapus data
        self.data_historis['tahun'].pop(indeks)
        self.data_historis['inflasi'].pop(indeks)
        self.data_historis['suku_bunga'].pop(indeks)
        self.data_historis['indeks_usd'].pop(indeks)
        self.data_historis['harga_emas'].pop(indeks)

        # Memperbarui tampilan
        self.perbarui_tampilan_data()

    def jalankan_prediksi(self):
        """Menjalankan perhitungan prediksi"""
        try:
            # Mendapatkan parameter input
            tahun = int(self.var_tahun.get())
            inflasi = float(self.var_inflasi.get())
            suku_bunga = float(self.var_suku_bunga.get())
            indeks_usd = float(self.var_usd.get())

            # Memperbarui parameter prediksi
            self.parameter_prediksi['tahun'] = tahun
            self.parameter_prediksi['inflasi'] = inflasi
            self.parameter_prediksi['suku_bunga'] = suku_bunga
            self.parameter_prediksi['indeks_usd'] = indeks_usd

            # Menyiapkan matriks untuk perhitungan SVD
            M = np.array([
                self.data_historis['inflasi'],
                self.data_historis['suku_bunga'],
                self.data_historis['indeks_usd']
            ]).T

            y = np.array(self.data_historis['harga_emas'])

            Mx = np.array([inflasi, suku_bunga, indeks_usd])

            # Menjalankan prediksi SVD
            # Langkah 1: Hitung SVD dari matriks M
            U, S, VT = np.linalg.svd(M, full_matrices=False)
            V = VT.T

            # Langkah 2: Hitung U^T * y
            UT_y = U.T @ y

            # Langkah 3: Hitung Sigma^+ (pseudo-inverse) * (U^T * y)
            S_inv = np.zeros_like(S)
            for i in range(len(S)):
                if S[i] > 0:
                    S_inv[i] = 1 / S[i]

            Sigma_plus_UT_y = S_inv * UT_y  # Perkalian elemen per elemen

            # Langkah 4: Hitung omega = V * Sigma^+ * (U^T * y)
            omega = V @ Sigma_plus_UT_y

            # Langkah 5: Hitung prediksi menggunakan Mx * omega
            harga_prediksi = Mx @ omega

            # Menyimpan hasil
            self.hasil['harga_prediksi'] = harga_prediksi
            self.hasil['komponen_svd'] = {
                'U': U,
                'S': S,
                'V': V
            }
            self.hasil['vektor_omega'] = omega

            # Menghitung galat dibandingkan dengan perhitungan manual
            galat = abs(harga_prediksi - self.hasil['perhitungan_manual']) / self.hasil['perhitungan_manual'] * 100
            self.hasil['persentase_galat'] = galat

            # Memperbarui UI dengan hasil
            self.label_hasil.config(text=f"${harga_prediksi:.2f} USD")

            # Memperbarui teks detail
            self.teks_detail.config(state=tk.NORMAL)
            self.teks_detail.delete(1.0, tk.END)
            detail = f"""Tahun: {tahun}
Parameter Prediksi:
- Inflasi: {inflasi}%
- Suku Bunga: {suku_bunga}%
- Indeks USD: {indeks_usd}

Harga Emas Prediksi: ${harga_prediksi:.4f} USD
Perhitungan Manual: ${self.hasil['perhitungan_manual']:.4f} USD
Galat: {galat:.8f}%

Perhitungan selesai pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.teks_detail.insert(tk.END, detail)
            self.teks_detail.config(state=tk.DISABLED)

            # Memperbarui teks komponen SVD
            self.teks_svd.config(state=tk.NORMAL)
            self.teks_svd.delete(1.0, tk.END)
            teks_svd = f"""Komponen SVD:

U (Vektor Singular Kiri):
{np.array2string(U, precision=6, suppress_small=True)}

Σ (Nilai Singular):
{np.array2string(S, precision=6, suppress_small=True)}

V (Vektor Singular Kanan):
{np.array2string(V, precision=6, suppress_small=True)}

U^T * y:
{np.array2string(UT_y, precision=6, suppress_small=True)}

Σ⁺ * (U^T * y):
{np.array2string(Sigma_plus_UT_y, precision=6, suppress_small=True)}

ω (Vektor Omega):
{np.array2string(omega, precision=6, suppress_small=True)}
"""
            self.teks_svd.insert(tk.END, teks_svd)
            self.teks_svd.config(state=tk.DISABLED)

            # Memperbarui analisis galat
            self.teks_galat.config(state=tk.NORMAL)
            self.teks_galat.delete(1.0, tk.END)
            teks_galat = f"""Analisis Galat:

Nilai prediksi: ${harga_prediksi:.7f} USD
Perhitungan manual: ${self.hasil['perhitungan_manual']:.7f} USD
Selisih absolut: ${abs(harga_prediksi - self.hasil['perhitungan_manual']):.7f} USD
Galat relatif: {galat:.10f}%

Galat ini kemungkinan disebabkan oleh perbedaan presisi floating-point antara perhitungan manual dan implementasi numpy.
"""
            self.teks_galat.insert(tk.END, teks_galat)
            self.teks_galat.config(state=tk.DISABLED)

            # Memperbarui visualisasi prediksi
            self.perbarui_visualisasi_prediksi()

            messagebox.showinfo("Prediksi Selesai",
                                f"Harga emas prediksi untuk tahun {tahun}: ${harga_prediksi:.2f} USD")

        except ValueError:
            messagebox.showerror("Kesalahan Input", "Mohon masukkan nilai numerik yang valid")

        except Exception as e:
            messagebox.showerror("Kesalahan Perhitungan", f"Terjadi kesalahan: {str(e)}")

    def jalankan_analisis_sensitivitas(self):
        """Menjalankan analisis sensitivitas pada parameter yang dipilih"""
        if self.hasil['harga_prediksi'] is None:
            messagebox.showinfo("Prediksi Diperlukan", "Mohon jalankan prediksi terlebih dahulu")
            return

        try:
            # Mendapatkan parameter
            param = self.param_sensitivitas.get()
            rentang_persen = float(self.rentang_sensitivitas.get())
            langkah = int(self.langkah_sensitivitas.get())

            # Mendapatkan nilai dasar
            nilai_dasar = self.parameter_prediksi[param]

            # Menghitung rentang
            nilai_min = nilai_dasar * (1 - rentang_persen / 100)
            nilai_max = nilai_dasar * (1 + rentang_persen / 100)

            # Menghasilkan nilai
            nilai = np.linspace(nilai_min, nilai_max, langkah)

            # Menjalankan prediksi untuk setiap nilai
            harga = []

            for val in nilai:
                # Menyiapkan parameter uji
                inflasi = self.parameter_prediksi['inflasi']
                suku_bunga = self.parameter_prediksi['suku_bunga']
                indeks_usd = self.parameter_prediksi['indeks_usd']

                if param == 'inflasi':
                    inflasi = val
                elif param == 'suku_bunga':
                    suku_bunga = val
                elif param == 'indeks_usd':
                    indeks_usd = val

                # Membuat matriks
                Mx = np.array([inflasi, suku_bunga, indeks_usd])

                # Menghitung prediksi
                harga_prediksi = Mx @ self.hasil['vektor_omega']
                harga.append(harga_prediksi)

            # Menyimpan hasil
            self.hasil['analisis_sensitivitas'] = {
                'param': param,
                'nilai': nilai,
                'harga': harga
            }

            # Memperbarui visualisasi
            self.perbarui_analisis_sensitivitas()

        except ValueError:
            messagebox.showerror("Kesalahan Input", "Mohon masukkan nilai numerik yang valid")

        except Exception as e:
            messagebox.showerror("Kesalahan Analisis", f"Terjadi kesalahan: {str(e)}")

    def impor_data(self):
        """Mengimpor data dari file CSV"""
        file_path = filedialog.askopenfilename(
            filetypes=[("File CSV", "*.csv"), ("Semua file", "*.*")]
        )

        if not file_path:
            return

        try:
            # Membaca file CSV
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Lewati baris header

                # Menyiapkan data
                tahun = []
                inflasi = []
                suku_bunga = []
                indeks_usd = []
                harga_emas = []

                for row in reader:
                    tahun.append(int(row[0]))
                    inflasi.append(float(row[1]))
                    suku_bunga.append(float(row[2]))
                    indeks_usd.append(float(row[3]))
                    harga_emas.append(float(row[4]))

                # Memperbarui data historis
                self.data_historis = {
                    'tahun': tahun,
                    'inflasi': inflasi,
                    'suku_bunga': suku_bunga,
                    'indeks_usd': indeks_usd,
                    'harga_emas': harga_emas
                }

                # Memperbarui tampilan
                self.perbarui_tampilan_data()

                messagebox.showinfo("Impor Berhasil", f"Berhasil mengimpor data dari {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Kesalahan Impor", f"Terjadi kesalahan: {str(e)}")

    def ekspor_hasil(self):
        """Mengekspor hasil prediksi ke file CSV"""
        if self.hasil['harga_prediksi'] is None:
            messagebox.showinfo("Prediksi Diperlukan", "Mohon jalankan prediksi terlebih dahulu")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("File CSV", "*.csv"), ("Semua file", "*.*")],
            initialfile=f"prediksi_emas_{self.parameter_prediksi['tahun']}.csv"
        )

        if not file_path:
            return

        try:
            # Membuat data untuk diekspor
            data = [
                ["Hasil Prediksi Harga Emas"],
                ["Dibuat pada", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                [""],
                ["Data Historis"],
                ["Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD", "Harga Emas (USD)"]
            ]

            for i in range(len(self.data_historis['tahun'])):
                data.append([
                    self.data_historis['tahun'][i],
                    self.data_historis['inflasi'][i],
                    self.data_historis['suku_bunga'][i],
                    self.data_historis['indeks_usd'][i],
                    self.data_historis['harga_emas'][i]
                ])

            data.extend([
                [""],
                ["Parameter Prediksi"],
                ["Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD"],
                [
                    self.parameter_prediksi['tahun'],
                    self.parameter_prediksi['inflasi'],
                    self.parameter_prediksi['suku_bunga'],
                    self.parameter_prediksi['indeks_usd']
                ],
                [""],
                ["Hasil Prediksi"],
                ["Harga Emas Prediksi (USD)", self.hasil['harga_prediksi']],
                ["Perhitungan Manual (USD)", self.hasil['perhitungan_manual']],
                ["Galat (%)", self.hasil['persentase_galat']],
                [""],
                ["Komponen SVD"],
                ["U (Vektor Singular Kiri)"],
            ])

            # Tambahkan matriks U
            for row in self.hasil['komponen_svd']['U']:
                data.append(row.tolist())

            data.extend([
                [""],
                ["Σ (Nilai Singular)"],
                self.hasil['komponen_svd']['S'].tolist(),
                [""],
                ["V (Vektor Singular Kanan)"],
            ])

            # Tambahkan matriks V
            for row in self.hasil['komponen_svd']['V']:
                data.append(row.tolist())

            data.extend([
                [""],
                ["ω (Vektor Omega)"],
                self.hasil['vektor_omega'].tolist()
            ])

            # Tulis ke CSV
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)

            messagebox.showinfo("Ekspor Berhasil", f"Berhasil mengekspor hasil ke {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Kesalahan Ekspor", f"Terjadi kesalahan: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    aplikasi = AplikasiPrediksiHargaEmas(root)
    root.mainloop()