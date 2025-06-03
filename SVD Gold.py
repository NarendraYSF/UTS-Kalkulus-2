"""
===============================================================================
                            Dokumentasi Kode
===============================================================================

Judul: Aplikasi Prediksi Harga Emas dengan SVD
Deskripsi:
    Skrip ini adalah aplikasi yang dirancang untuk memprediksi harga emas berdasarkan
    analisis historis dan berbagai faktor ekonomi. Aplikasi ini menggunakan Singular
    Value Decomposition (SVD) untuk menganalisis data historis dan menghitung prediksi
    harga emas di masa depan berdasarkan variabel-variabel seperti inflasi, suku bunga,
    dan indeks USD.

    Fitur utama aplikasi meliputi:
    - Pengolahan data historis untuk memasukkan variabel ekonomi yang relevan
    - Prediksi harga emas menggunakan berbagai model regresi
    - Analisis sensitivitas terhadap perubahan parameter ekonomi
    - Visualisasi data dan hasil analisis dalam bentuk grafik dan tabel
    - Kemampuan untuk memperbarui data secara dinamis dan melakukan optimasi model

Penulis: Narendra Yusuf 未来
Tanggal: May 20 2025
Versi: 2.1

===============================================================================
                            Deskripsi Data
===============================================================================

Data input yang digunakan terdiri dari informasi historis terkait inflasi, suku bunga,
indeks USD, dan harga emas:
    - 'tahun': Tahun data historis
    - 'inflasi': Tingkat inflasi per tahun
    - 'suku_bunga': Tingkat suku bunga yang berlaku
    - 'indeks_usd': Indeks kekuatan dolar Amerika Serikat
    - 'harga_emas': Harga emas per ons dalam USD

Data ini digunakan untuk membangun model prediksi harga emas, serta untuk analisis
sensitivitas terhadap perubahan parameter ekonomi.

===============================================================================
                            Ikhtisar Fungsionalitas
===============================================================================

1. **Pengolahan Data Historis**:
    - Data historis digunakan untuk menghitung harga emas berdasarkan berbagai faktor ekonomi.
    - Aplikasi menyediakan antarmuka untuk memuat dan memperbarui data historis.

2. **Model Prediksi**:
    - Berdasarkan data historis, aplikasi membangun model prediksi harga emas menggunakan regresi linier dan model SVD.
    - Pengguna dapat memilih model untuk digunakan dalam prediksi harga emas masa depan.

3. **Analisis Sensitivitas**:
    - Aplikasi menyediakan analisis sensitivitas untuk mengevaluasi bagaimana perubahan dalam parameter ekonomi (inflasi, suku bunga, indeks USD) mempengaruhi prediksi harga emas.
    - Hasil analisis sensitivitas dapat divisualisasikan untuk pemahaman lebih mendalam.

4. **Optimasi dan Validasi Model**:
    - Aplikasi menggunakan teknik validasi silang dan perbandingan model untuk memilih model terbaik yang memberikan hasil prediksi yang akurat.
    - Model yang digunakan dapat disesuaikan dengan parameter yang berbeda.

5. **Visualisasi Hasil**:
    - Grafik interaktif digunakan untuk memvisualisasikan data historis, prediksi harga emas, serta hasil analisis sensitivitas dan model perbandingan.

6. **Antarmuka Pengguna (GUI)**:
    - Antarmuka grafis (GUI) yang intuitif memungkinkan pengguna untuk mengimpor data, memilih model prediksi, dan memvisualisasikan hasil analisis.
    - Pengguna dapat memilih tema antarmuka (terang atau gelap) sesuai preferensi mereka.

7. **Fitur Auto-Save**:
    - Aplikasi dapat menyimpan pengaturan dan hasil secara otomatis pada interval waktu tertentu untuk memastikan data tidak hilang.

8. **Pengaturan Tema dan Bahasa**:
    - Aplikasi memungkinkan pengaturan tema (gelap atau terang) serta bahasa antarmuka yang dapat disesuaikan.

===============================================================================
                            Pembagian Kode
===============================================================================

1. **Kelas EnhancedAplikasiPrediksiHargaEmas**:
    - Kelas utama yang menangani antarmuka pengguna (GUI) dan logika aplikasi. Di dalamnya termasuk
      metode untuk memuat data, memilih model, menjalankan prediksi, serta menampilkan hasil evaluasi dan
      visualisasi data.

2. **Fungsi untuk Menangani Data**:
    - Fungsi yang mengimpor dan mengekspor data, serta memperbarui dan mengelola data historis yang digunakan untuk
      pelatihan model prediksi.

3. **Fungsi Prediksi**:
    - Fungsi untuk menjalankan prediksi harga emas berdasarkan input variabel ekonomi dan model yang dipilih.

4. **Fungsi Analisis Sensitivitas**:
    - Fungsi untuk melakukan analisis sensitivitas terhadap perubahan parameter ekonomi, serta menghasilkan
      visualisasi dan laporan analisis.

5. **Fungsi Optimasi Model**:
    - Fungsi untuk membandingkan berbagai model dan memilih yang terbaik menggunakan teknik validasi silang (cross-validation) dan perbandingan model.

6. **Fungsi Visualisasi**:
    - Fungsi untuk membuat grafik dan plot yang menggambarkan data historis, hasil prediksi, dan analisis sensitivitas.

7. **Pengaturan Aplikasi**:
    - Aplikasi menyediakan pengaturan untuk memilih tema (terang atau gelap), mengaktifkan fitur auto-save, dan memilih bahasa antarmuka.

===============================================================================
                            Instruksi Penggunaan
===============================================================================

1. **Menjalankan Aplikasi**:
    - Buka aplikasi dan pilih data historis yang akan digunakan untuk pelatihan model prediksi harga emas.
    - Pilih model yang ingin digunakan untuk prediksi, baik regresi linier atau model berbasis SVD.

2. **Melakukan Prediksi**:
    - Masukkan parameter ekonomi untuk tahun yang diinginkan (misalnya, inflasi, suku bunga, dan indeks USD) untuk melakukan prediksi harga emas.
    - Hasil prediksi akan ditampilkan di antarmuka aplikasi, dan grafik visualisasi akan menunjukkan hasilnya.

3. **Analisis Sensitivitas**:
    - Gunakan tab analisis untuk mengevaluasi bagaimana perubahan parameter ekonomi mempengaruhi prediksi harga emas.
    - Hasil analisis sensitivitas akan disajikan dalam bentuk tabel dan grafik.

4. **Optimasi Model**:
    - Aplikasi secara otomatis akan mengoptimalkan model menggunakan teknik validasi silang untuk memilih model terbaik berdasarkan metrik evaluasi yang relevan.

5. **Visualisasi Data**:
    - Gunakan grafik interaktif untuk melihat data historis, hasil prediksi, serta analisis sensitivitas dalam bentuk plot.
    - Anda juga dapat mengekspor grafik dalam format yang diinginkan untuk presentasi atau laporan.

6. **Pengaturan Tema**:
    - Aplikasi memungkinkan Anda untuk memilih antara tema terang atau gelap untuk pengalaman pengguna yang lebih baik.

7. **Auto-Save**:
    - Aplikasi secara otomatis menyimpan hasil dan pengaturan pada interval waktu tertentu (default setiap 5 menit), yang dapat disesuaikan melalui pengaturan aplikasi.

===============================================================================
"""
#region
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import csv
import os
import json
from datetime import datetime, timedelta
import io
from PIL import Image, ImageTk
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from scipy import stats
from scipy.signal import correlate
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
#endregion

warnings.filterwarnings('ignore')


class EnhancedAplikasiPrediksiHargaEmas:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Prediksi Harga Emas - Versi Ditingkatkan v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f5f5f5")

        # Konfigurasi tema
        self.tema_gelap = False
        self.warna_tema = {
            'terang': {
                'bg': '#f5f5f5',
                'fg': '#000000',
                'select_bg': '#0078d4',
                'select_fg': '#ffffff'
            },
            'gelap': {
                'bg': '#2d2d2d',
                'fg': '#ffffff',
                'select_bg': '#404040',
                'select_fg': '#ffffff'
            }
        }

        # Cache untuk optimasi
        self.cache_perhitungan = {}
        self.cache_visualisasi = {}

        # Status aplikasi
        self.sedang_memproses = False
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Siap")

        # Thread pool untuk operasi paralel
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Pengaturan aplikasi
        self.pengaturan = {
            'auto_save': True,
            'tema': 'terang',
            'bahasa': 'indonesia',
            'cache_enabled': True,
            'parallel_processing': True
        }

        try:
            self.root.iconbitmap("gold_icon.ico")
        except:
            pass

        # Data default dari makalah dengan data tambahan
        self.data_historis = {
            'tahun': [2020, 2021, 2022, 2023, 2024],
            'inflasi': [2.75, 3.12, 3.36, 1.80, 2.09],
            'suku_bunga': [3.5, 3.75, 4.0, 5.8125, 6.1042],
            'indeks_usd': [100.7, 95.6, 104.0558, 103.4642, 104.46],
            'harga_emas': [1773.3, 1807.2, 1806.9667, 1962.2, 2416.4217]
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
            'analisis_sensitivitas': {},
            'confidence_interval': None,
            'model_comparison': {},
            'cross_validation_scores': None,
            'feature_importance': None,
            'residuals': None,
            'y_train_pred': None
        }

        # Model tambahan - FIXED: Keys now match radio button values
        self.model_alternatif = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Membuat elemen UI utama
        self.buat_ui()
        self.buat_menu()
        self.buat_toolbar()
        self.buat_status_bar()

        # Memuat pengaturan
        self.muat_pengaturan()

        # Memuat data awal dan memperbarui UI
        self.perbarui_tampilan_data()

        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Auto-save timer
        if self.pengaturan['auto_save']:
            self.setup_auto_save()

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.impor_data())
        self.root.bind('<Control-s>', lambda e: self.ekspor_hasil())
        self.root.bind('<Control-r>', lambda e: self.jalankan_prediksi())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<Control-t>', lambda e: self.toggle_tema())

    def setup_auto_save(self):
        """Setup auto-save timer"""

        def auto_save():
            if self.pengaturan['auto_save']:
                self.simpan_pengaturan()
                self.root.after(300000, auto_save)  # 5 menit

        self.root.after(300000, auto_save)

    def buat_menu(self):
        """Membuat menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Impor Data (Ctrl+O)", command=self.impor_data)
        file_menu.add_command(label="Ekspor Hasil (Ctrl+S)", command=self.ekspor_hasil)
        file_menu.add_separator()
        file_menu.add_command(label="Ekspor Laporan PDF", command=self.ekspor_laporan_pdf)
        file_menu.add_command(label="Ekspor Grafik", command=self.ekspor_grafik)
        file_menu.add_separator()
        file_menu.add_command(label="Keluar", command=self.root.quit)

        # Menu Edit
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Refresh Data (F5)", command=self.refresh_data)
        edit_menu.add_command(label="Reset Cache", command=self.reset_cache)
        edit_menu.add_separator()
        edit_menu.add_command(label="Pengaturan", command=self.buka_pengaturan)

        # Menu Analisis
        analisis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analisis", menu=analisis_menu)
        analisis_menu.add_command(label="Jalankan Prediksi (Ctrl+R)", command=self.jalankan_prediksi)
        analisis_menu.add_command(label="Analisis Komprehensif", command=self.analisis_komprehensif)
        analisis_menu.add_command(label="Backtesting", command=self.jalankan_backtesting)
        analisis_menu.add_command(label="Cross Validation", command=self.jalankan_cross_validation)

        # Menu Visualisasi
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisasi", menu=viz_menu)
        viz_menu.add_command(label="Grafik Interaktif", command=self.buka_grafik_interaktif)
        viz_menu.add_command(label="Surface Plot 3D", command=self.buka_surface_plot_3d)
        viz_menu.add_command(label="Heatmap Korelasi", command=self.buka_heatmap_korelasi)

        # Menu Tema
        tema_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tema", menu=tema_menu)
        tema_menu.add_command(label="Tema Terang", command=lambda: self.ubah_tema('terang'))
        tema_menu.add_command(label="Tema Gelap", command=lambda: self.ubah_tema('gelap'))
        tema_menu.add_command(label="Toggle Tema (Ctrl+T)", command=self.toggle_tema)

        # Menu Bantuan
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bantuan", menu=help_menu)
        help_menu.add_command(label="Panduan Penggunaan", command=self.buka_panduan)
        help_menu.add_command(label="Tentang Aplikasi", command=self.buka_tentang)

    def buat_toolbar(self):
        """Membuat toolbar"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(fill=tk.X, padx=5, pady=2)

        # Tombol toolbar
        ttk.Button(self.toolbar, text="📁 Impor", command=self.impor_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="💾 Ekspor", command=self.ekspor_hasil).pack(side=tk.LEFT, padx=2)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(self.toolbar, text="🔍 Prediksi", command=self.jalankan_prediksi).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="📊 Analisis", command=self.analisis_komprehensif).pack(side=tk.LEFT, padx=2)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(self.toolbar, text="🔄 Refresh", command=self.refresh_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="⚙️ Pengaturan", command=self.buka_pengaturan).pack(side=tk.LEFT, padx=2)

    def buat_status_bar(self):
        """Membuat status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Status label
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)

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
        self.tab_model_comparison = ttk.Frame(self.tab_control)
        self.tab_backtesting = ttk.Frame(self.tab_control)
        self.tab_teori = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_data, text='📊 Data Historis')
        self.tab_control.add(self.tab_prediksi, text='🔮 Prediksi')
        self.tab_control.add(self.tab_analisis, text='📈 Analisis Lanjutan')
        self.tab_control.add(self.tab_model_comparison, text='🔬 Perbandingan Model')
        self.tab_control.add(self.tab_backtesting, text='⏮️ Backtesting')
        self.tab_control.add(self.tab_teori, text='📚 Teori & Panduan')

        self.tab_control.pack(expand=1, fill=tk.BOTH)

        # Mengisi setiap tab dengan konten
        self.buat_tab_data()
        self.buat_tab_prediksi()
        self.buat_tab_analisis()
        self.buat_tab_model_comparison()
        self.buat_tab_backtesting()
        self.buat_tab_teori()

    def buat_header(self):
        """Membuat header aplikasi"""
        bingkai_header = ttk.Frame(self.root)
        bingkai_header.pack(fill=tk.X, pady=10)

        label_judul = ttk.Label(
            bingkai_header,
            text="🏆 Aplikasi Prediksi Harga Emas - Versi Ditingkatkan",
            font=("Arial", 18, "bold")
        )
        label_judul.pack()

        label_subjudul = ttk.Label(
            bingkai_header,
            text="Menggunakan SVD, Machine Learning & Analisis Statistik Lanjutan",
            font=("Arial", 12)
        )
        label_subjudul.pack()

    def buat_tab_data(self):
        """Membuat tab data historis dengan fitur enhanced"""
        bingkai = ttk.Frame(self.tab_data)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Data table dan kontrol
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri ===
        # Data controls
        kontrol_frame = ttk.LabelFrame(frame_kiri, text="Kontrol Data")
        kontrol_frame.pack(fill=tk.X, pady=5)

        # Tombol kontrol dalam grid
        ttk.Button(kontrol_frame, text="➕ Tambah", command=self.tambah_baris_data).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="✏️ Edit", command=self.edit_baris_data).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="🗑️ Hapus", command=self.hapus_baris_data).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="📥 Impor", command=self.impor_data).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="📤 Ekspor", command=self.ekspor_data).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="🔄 Refresh", command=self.refresh_data).grid(row=1, column=2, padx=2, pady=2)

        # Tabel data dengan scrollbar
        table_frame = ttk.LabelFrame(frame_kiri, text="Data Historis")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview dengan scrollbar
        tree_frame = ttk.Frame(table_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pohon_data = ttk.Treeview(
            tree_frame,
            columns=("Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD", "Harga Emas (USD)"),
            show='headings'
        )

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.pohon_data.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.pohon_data.xview)
        self.pohon_data.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.pohon_data.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Setup kolom
        for col in ("Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD", "Harga Emas (USD)"):
            self.pohon_data.heading(col, text=col, command=lambda c=col: self.sort_treeview(c))
            self.pohon_data.column(col, width=100, anchor=tk.CENTER)

        # Statistik data
        stats_frame = ttk.LabelFrame(frame_kiri, text="Statistik Data")
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_text = tk.Text(stats_frame, height=8, width=30)
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan ===
        # Kontrol visualisasi
        viz_control_frame = ttk.LabelFrame(frame_kanan, text="Kontrol Visualisasi")
        viz_control_frame.pack(fill=tk.X, pady=5)

        # Row 1
        row1 = ttk.Frame(viz_control_frame)
        row1.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(row1, text="Jenis Grafik:").pack(side=tk.LEFT, padx=5)
        self.jenis_plot = tk.StringVar(value="line")
        combo_plot = ttk.Combobox(row1, textvariable=self.jenis_plot,
                                  values=["line", "bar", "scatter", "area"], width=10, state="readonly")
        combo_plot.pack(side=tk.LEFT, padx=5)
        combo_plot.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_data)

        ttk.Label(row1, text="Variabel:").pack(side=tk.LEFT, padx=5)
        self.var_visualisasi = tk.StringVar(value="semua")
        combo_var = ttk.Combobox(row1, textvariable=self.var_visualisasi,
                                 values=["semua", "harga_emas", "inflasi", "suku_bunga", "indeks_usd"],
                                 width=12, state="readonly")
        combo_var.pack(side=tk.LEFT, padx=5)
        combo_var.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_data)

        # Row 2
        row2 = ttk.Frame(viz_control_frame)
        row2.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(row2, text="🔄 Perbarui", command=self.perbarui_visualisasi_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="💾 Simpan Grafik", command=self.simpan_grafik_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="🔍 Zoom Reset", command=self.reset_zoom_data).pack(side=tk.LEFT, padx=5)

        # Area visualisasi dengan toolbar
        viz_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Data")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.gambar_data = Figure(figsize=(10, 8), dpi=100)
        self.kanvas_data = FigureCanvasTkAgg(self.gambar_data, viz_frame)

        # Toolbar navigasi matplotlib
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=2)

        self.navbar_data = NavigationToolbar2Tk(self.kanvas_data, toolbar_frame)
        self.navbar_data.update()

        self.kanvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def buat_tab_prediksi(self):
        """Membuat tab prediksi dengan fitur enhanced"""
        bingkai = ttk.Frame(self.tab_prediksi)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Parameter dan kontrol
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Hasil dan visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri ===
        # Parameter prediksi dengan tooltips
        param_frame = ttk.LabelFrame(frame_kiri, text="Parameter Prediksi")
        param_frame.pack(fill=tk.X, pady=5, padx=5)

        # Tahun
        ttk.Label(param_frame, text="Tahun:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_tahun = tk.StringVar(value=str(self.parameter_prediksi['tahun']))
        entry_tahun = ttk.Entry(param_frame, textvariable=self.var_tahun, width=10)
        entry_tahun.grid(row=0, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_tahun, "Masukkan tahun untuk prediksi (contoh: 2025)")

        # Inflasi
        ttk.Label(param_frame, text="Inflasi (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_inflasi = tk.StringVar(value=str(self.parameter_prediksi['inflasi']))
        entry_inflasi = ttk.Entry(param_frame, textvariable=self.var_inflasi, width=10)
        entry_inflasi.grid(row=1, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_inflasi, "Tingkat inflasi yang diharapkan dalam persen")

        # Suku Bunga
        ttk.Label(param_frame, text="Suku Bunga (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_suku_bunga = tk.StringVar(value=str(self.parameter_prediksi['suku_bunga']))
        entry_suku_bunga = ttk.Entry(param_frame, textvariable=self.var_suku_bunga, width=10)
        entry_suku_bunga.grid(row=2, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_suku_bunga, "Suku bunga acuan dalam persen")

        # Indeks USD
        ttk.Label(param_frame, text="Indeks USD:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_usd = tk.StringVar(value=str(self.parameter_prediksi['indeks_usd']))
        entry_usd = ttk.Entry(param_frame, textvariable=self.var_usd, width=10)
        entry_usd.grid(row=3, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_usd, "Indeks kekuatan Dollar AS")

        # Tombol prediksi
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="🔮 Hitung Prediksi",
                   command=self.jalankan_prediksi_async).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📊 Analisis Komprehensif",
                   command=self.analisis_komprehensif).pack(side=tk.LEFT, padx=2)

        # Model selection
        model_frame = ttk.LabelFrame(frame_kiri, text="Pilihan Model")
        model_frame.pack(fill=tk.X, pady=5, padx=5)

        self.model_terpilih = tk.StringVar(value="svd")
        models = [
            ("SVD + Least Squares", "svd"),
            ("Linear Regression", "linear"),
            ("Ridge Regression", "ridge"),
            ("Lasso Regression", "lasso"),
            ("Random Forest", "random_forest")
        ]

        for i, (text, value) in enumerate(models):
            ttk.Radiobutton(model_frame, text=text, variable=self.model_terpilih,
                            value=value).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

        # Opsi lanjutan
        advanced_frame = ttk.LabelFrame(frame_kiri, text="Opsi Lanjutan")
        advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        self.var_confidence = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Hitung Confidence Interval",
                        variable=self.var_confidence).pack(anchor=tk.W, padx=5, pady=2)

        self.var_feature_importance = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Analisis Feature Importance",
                        variable=self.var_feature_importance).pack(anchor=tk.W, padx=5, pady=2)

        self.var_residual_analysis = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Analisis Residual",
                        variable=self.var_residual_analysis).pack(anchor=tk.W, padx=5, pady=2)

        # SVD Components
        svd_frame = ttk.LabelFrame(frame_kiri, text="Komponen SVD")
        svd_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.teks_svd = tk.Text(svd_frame, height=12, width=40)
        svd_scroll = ttk.Scrollbar(svd_frame, orient=tk.VERTICAL, command=self.teks_svd.yview)
        self.teks_svd.configure(yscrollcommand=svd_scroll.set)

        self.teks_svd.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        svd_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan ===
        # Hasil prediksi
        hasil_frame = ttk.LabelFrame(frame_kanan, text="Hasil Prediksi")
        hasil_frame.pack(fill=tk.X, pady=5, padx=5)

        # Hasil utama dengan styling
        self.label_hasil = ttk.Label(hasil_frame, text="Jalankan prediksi untuk melihat hasil",
                                     font=("Arial", 24, "bold"), foreground="blue")
        self.label_hasil.pack(pady=10)

        # Detail hasil
        detail_notebook = ttk.Notebook(hasil_frame)
        detail_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab detail umum
        tab_detail = ttk.Frame(detail_notebook)
        detail_notebook.add(tab_detail, text="Detail")

        self.teks_detail = tk.Text(tab_detail, height=8, width=60)
        detail_scroll = ttk.Scrollbar(tab_detail, orient=tk.VERTICAL, command=self.teks_detail.yview)
        self.teks_detail.configure(yscrollcommand=detail_scroll.set)

        self.teks_detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab metrics
        tab_metrics = ttk.Frame(detail_notebook)
        detail_notebook.add(tab_metrics, text="Metrics")

        self.teks_metrics = tk.Text(tab_metrics, height=8, width=60)
        metrics_scroll = ttk.Scrollbar(tab_metrics, orient=tk.VERTICAL, command=self.teks_metrics.yview)
        self.teks_metrics.configure(yscrollcommand=metrics_scroll.set)

        self.teks_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Visualisasi prediksi
        viz_pred_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Prediksi")
        viz_pred_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        # Kontrol visualisasi prediksi
        pred_control = ttk.Frame(viz_pred_frame)
        pred_control.pack(fill=tk.X, pady=2)

        ttk.Label(pred_control, text="Tampilan:").pack(side=tk.LEFT, padx=5)
        self.pred_view = tk.StringVar(value="trend")
        combo_pred = ttk.Combobox(pred_control, textvariable=self.pred_view,
                                  values=["trend", "confidence", "residuals", "comparison"],
                                  width=12, state="readonly")
        combo_pred.pack(side=tk.LEFT, padx=5)
        combo_pred.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_prediksi)

        ttk.Button(pred_control, text="🔄 Perbarui",
                   command=self.perbarui_visualisasi_prediksi).pack(side=tk.LEFT, padx=5)

        self.gambar_pred = Figure(figsize=(10, 6), dpi=100)
        self.kanvas_pred = FigureCanvasTkAgg(self.gambar_pred, viz_pred_frame)

        # Toolbar navigasi
        toolbar_pred_frame = ttk.Frame(viz_pred_frame)
        toolbar_pred_frame.pack(fill=tk.X, padx=5, pady=2)

        self.navbar_pred = NavigationToolbar2Tk(self.kanvas_pred, toolbar_pred_frame)
        self.navbar_pred.update()

        self.kanvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def buat_tab_analisis(self):
        """Membuat tab analisis lanjutan dengan fitur komprehensif"""
        bingkai = ttk.Frame(self.tab_analisis)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Kontrol analisis
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Hasil dan visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri - Kontrol ===
        # Pilihan jenis analisis
        analisis_frame = ttk.LabelFrame(frame_kiri, text="Jenis Analisis")
        analisis_frame.pack(fill=tk.X, pady=5, padx=5)

        self.jenis_analisis = tk.StringVar(value="korelasi")
        analisis_options = [
            ("Analisis Korelasi", "korelasi"),
            ("Analisis Sensitivitas", "sensitivitas"),
            ("Analisis Trend", "trend"),
            ("Analisis Outlier", "outlier"),
            ("Analisis Time Series", "timeseries"),
            ("Statistical Testing", "statistical"),
            ("Monte Carlo Simulation", "montecarlo"),
            ("Feature Analysis", "features")
        ]

        for i, (text, value) in enumerate(analisis_options):
            ttk.Radiobutton(analisis_frame, text=text, variable=self.jenis_analisis,
                            value=value, command=self.update_analisis_options).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)

        # Parameter analisis
        param_analisis_frame = ttk.LabelFrame(frame_kiri, text="Parameter Analisis")
        param_analisis_frame.pack(fill=tk.X, pady=5, padx=5)

        # Confidence level
        ttk.Label(param_analisis_frame, text="Confidence Level:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.confidence_level = tk.DoubleVar(value=0.95)
        confidence_spinbox = ttk.Spinbox(param_analisis_frame, from_=0.9, to=0.99,
                                         increment=0.01, textvariable=self.confidence_level, width=10)
        confidence_spinbox.grid(row=0, column=1, padx=5, pady=5)

        # Simulation runs (untuk Monte Carlo)
        ttk.Label(param_analisis_frame, text="Simulation Runs:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.sim_runs = tk.IntVar(value=1000)
        sim_spinbox = ttk.Spinbox(param_analisis_frame, from_=100, to=10000,
                                  increment=100, textvariable=self.sim_runs, width=10)
        sim_spinbox.grid(row=1, column=1, padx=5, pady=5)

        # Outlier threshold
        ttk.Label(param_analisis_frame, text="Outlier Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.outlier_threshold = tk.DoubleVar(value=2.0)
        outlier_spinbox = ttk.Spinbox(param_analisis_frame, from_=1.0, to=5.0,
                                      increment=0.1, textvariable=self.outlier_threshold, width=10)
        outlier_spinbox.grid(row=2, column=1, padx=5, pady=5)

        # Tombol kontrol
        control_frame = ttk.Frame(param_analisis_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(control_frame, text="🔍 Jalankan Analisis",
                   command=self.jalankan_analisis_lanjutan).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="📊 Export Hasil",
                   command=self.export_analisis_hasil).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🔄 Reset",
                   command=self.reset_analisis).pack(side=tk.LEFT, padx=2)

        # Hasil analisis teks
        hasil_frame = ttk.LabelFrame(frame_kiri, text="Hasil Analisis")
        hasil_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.teks_analisis = tk.Text(hasil_frame, wrap=tk.WORD, font=("Consolas", 9))
        scroll_analisis = ttk.Scrollbar(hasil_frame, orient=tk.VERTICAL, command=self.teks_analisis.yview)
        self.teks_analisis.configure(yscrollcommand=scroll_analisis.set)

        self.teks_analisis.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll_analisis.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan - Visualisasi ===
        # Kontrol visualisasi
        viz_control_frame = ttk.LabelFrame(frame_kanan, text="Kontrol Visualisasi")
        viz_control_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(viz_control_frame, text="Layout:").pack(side=tk.LEFT, padx=5)
        self.viz_layout = tk.StringVar(value="single")
        layout_combo = ttk.Combobox(viz_control_frame, textvariable=self.viz_layout,
                                    values=["single", "grid", "subplots"], width=10, state="readonly")
        layout_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(viz_control_frame, text="🔄 Refresh Viz",
                   command=self.refresh_analisis_viz).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_control_frame, text="💾 Save Plot",
                   command=self.save_analisis_plot).pack(side=tk.LEFT, padx=5)

        # Area visualisasi
        viz_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Analisis")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.gambar_analisis = Figure(figsize=(12, 8), dpi=100)
        self.kanvas_analisis = FigureCanvasTkAgg(self.gambar_analisis, viz_frame)

        # Toolbar navigasi
        toolbar_analisis_frame = ttk.Frame(viz_frame)
        toolbar_analisis_frame.pack(fill=tk.X, padx=5, pady=2)

        self.navbar_analisis = NavigationToolbar2Tk(self.kanvas_analisis, toolbar_analisis_frame)
        self.navbar_analisis.update()

        self.kanvas_analisis.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Storage untuk hasil analisis
        self.hasil_analisis = {
            'korelasi': None,
            'sensitivitas': None,
            'trend': None,
            'outlier': None,
            'timeseries': None,
            'statistical': None,
            'montecarlo': None,
            'features': None
        }

    def update_analisis_options(self):
        """Update opsi berdasarkan jenis analisis yang dipilih"""
        jenis = self.jenis_analisis.get()

        # Reset text
        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, f"Siap untuk analisis: {jenis.title()}")
        self.teks_analisis.config(state=tk.DISABLED)

    def jalankan_analisis_lanjutan(self):
        """Menjalankan analisis lanjutan berdasarkan pilihan"""
        jenis = self.jenis_analisis.get()

        self.update_progress(10, f"Memulai analisis {jenis}...")

        try:
            if jenis == "korelasi":
                self.analisis_korelasi()
            elif jenis == "sensitivitas":
                self.analisis_sensitivitas()
            elif jenis == "trend":
                self.analisis_trend()
            elif jenis == "outlier":
                self.analisis_outlier()
            elif jenis == "timeseries":
                self.analisis_timeseries()
            elif jenis == "statistical":
                self.analisis_statistical()
            elif jenis == "montecarlo":
                self.analisis_montecarlo()
            elif jenis == "features":
                self.analisis_features()

            self.update_progress(100, "Analisis selesai!")

        except Exception as e:
            messagebox.showerror("Error Analisis", f"Terjadi kesalahan: {str(e)}")
            self.reset_progress()

    def analisis_korelasi(self):
        """Analisis korelasi antar variabel"""
        self.update_progress(30, "Menghitung korelasi...")

        # Prepare data
        data = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd'],
            self.data_historis['harga_emas']
        ]).T

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)

        # Calculate p-values
        n = len(data)
        p_values = np.zeros_like(corr_matrix)

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i != j:
                    r = corr_matrix[i, j]
                    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
                    p_values[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        self.hasil_analisis['korelasi'] = {
            'matrix': corr_matrix,
            'p_values': p_values,
            'variables': ['Inflasi', 'Suku Bunga', 'Indeks USD', 'Harga Emas']
        }

        # Update display
        self.update_korelasi_display()
        self.plot_korelasi()

    def analisis_sensitivitas(self):
        """Analisis sensitivitas parameter terhadap prediksi"""
        self.update_progress(30, "Menghitung sensitivitas...")

        # Base prediction
        base_params = [
            self.parameter_prediksi['inflasi'],
            self.parameter_prediksi['suku_bunga'],
            self.parameter_prediksi['indeks_usd']
        ]

        # Calculate base prediction
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T
        y = np.array(self.data_historis['harga_emas'])

        base_pred = self.predict_with_params(M, y, base_params)

        # Sensitivity analysis - vary each parameter
        sensitivity_results = {}
        param_names = ['inflasi', 'suku_bunga', 'indeks_usd']
        variations = [-20, -10, -5, 5, 10, 20]  # percentage variations

        for i, param_name in enumerate(param_names):
            param_sensitivity = []
            param_values = []

            for var in variations:
                new_params = base_params.copy()
                new_params[i] = base_params[i] * (1 + var / 100)

                pred = self.predict_with_params(M, y, new_params)
                sensitivity = ((pred - base_pred) / base_pred) * 100

                param_sensitivity.append(sensitivity)
                param_values.append(new_params[i])

            sensitivity_results[param_name] = {
                'variations': variations,
                'values': param_values,
                'sensitivity': param_sensitivity,
                'base_value': base_params[i]
            }

        self.hasil_analisis['sensitivitas'] = {
            'base_prediction': base_pred,
            'results': sensitivity_results
        }

        self.update_sensitivitas_display()
        self.plot_sensitivitas()

    def analisis_trend(self):
        """Analisis trend data historis"""
        self.update_progress(30, "Menganalisis trend...")

        tahun = np.array(self.data_historis['tahun'])
        variables = ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']
        trend_results = {}

        for var in variables:
            data = np.array(self.data_historis[var])

            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(tahun, data)

            # Moving average
            if len(data) >= 3:
                moving_avg = np.convolve(data, np.ones(3) / 3, mode='valid')
            else:
                moving_avg = data

            # Growth rate
            growth_rates = []
            for i in range(1, len(data)):
                growth_rate = ((data[i] - data[i - 1]) / data[i - 1]) * 100
                growth_rates.append(growth_rate)

            trend_results[var] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'moving_avg': moving_avg,
                'growth_rates': growth_rates,
                'avg_growth': np.mean(growth_rates) if growth_rates else 0,
                'volatility': np.std(data)
            }

        self.hasil_analisis['trend'] = trend_results

        self.update_trend_display()
        self.plot_trend()

    def analisis_outlier(self):
        """Analisis outlier dalam data"""
        self.update_progress(30, "Mendeteksi outlier...")

        variables = ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']
        outlier_results = {}
        threshold = self.outlier_threshold.get()

        for var in variables:
            data = np.array(self.data_historis[var])

            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = np.where(z_scores > threshold)[0]

            # IQR method
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = np.where((data < lower_bound) | (data > upper_bound))[0]

            # Modified Z-score (MAD)
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            mad_outliers = np.where(np.abs(modified_z_scores) > 3.5)[0]

            outlier_results[var] = {
                'z_outliers': z_outliers,
                'z_scores': z_scores,
                'iqr_outliers': iqr_outliers,
                'mad_outliers': mad_outliers,
                'modified_z_scores': modified_z_scores,
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
                'statistics': {
                    'mean': np.mean(data),
                    'median': median,
                    'std': np.std(data),
                    'mad': mad
                }
            }

        self.hasil_analisis['outlier'] = outlier_results

        self.update_outlier_display()
        self.plot_outlier()

    def analisis_timeseries(self):
        """Analisis time series decomposition"""
        self.update_progress(30, "Menganalisis time series...")

        # Simple time series analysis for gold price
        harga_emas = np.array(self.data_historis['harga_emas'])
        tahun = np.array(self.data_historis['tahun'])

        # Linear trend
        slope, intercept = np.polyfit(range(len(harga_emas)), harga_emas, 1)
        trend = slope * np.arange(len(harga_emas)) + intercept

        # Detrended series
        detrended = harga_emas - trend

        # Simple seasonal component (if data length allows)
        if len(harga_emas) >= 4:
            # Use simple moving average for seasonal pattern
            seasonal = np.zeros_like(harga_emas)
            for i in range(len(harga_emas)):
                seasonal[i] = np.mean(detrended[max(0, i - 1):min(len(detrended), i + 2)])
        else:
            seasonal = np.zeros_like(harga_emas)

        # Residuals
        residual = harga_emas - trend - seasonal

        # Statistical tests
        # Augmented Dickey-Fuller test (simplified)
        # Just calculate some basic statistics for now
        stationarity_stats = {
            'mean': np.mean(harga_emas),
            'std': np.std(harga_emas),
            'trend_strength': abs(slope),
            'residual_std': np.std(residual)
        }

        self.hasil_analisis['timeseries'] = {
            'original': harga_emas,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'slope': slope,
            'intercept': intercept,
            'stationarity': stationarity_stats
        }

        self.update_timeseries_display()
        self.plot_timeseries()

    def analisis_statistical(self):
        """Statistical testing"""
        self.update_progress(30, "Menjalankan uji statistik...")

        variables = ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']
        statistical_results = {}

        for var in variables:
            data = np.array(self.data_historis[var])

            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(data)

            # Descriptive statistics
            desc_stats = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'var': np.var(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data)
            }

            # Confidence interval for mean
            confidence = self.confidence_level.get()
            ci = stats.t.interval(confidence, len(data) - 1,
                                  loc=np.mean(data),
                                  scale=stats.sem(data))

            statistical_results[var] = {
                'descriptive': desc_stats,
                'normality': {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                },
                'confidence_interval': ci
            }

        # Correlation significance tests
        corr_tests = {}
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i + 1:], i + 1):
                data1 = np.array(self.data_historis[var1])
                data2 = np.array(self.data_historis[var2])

                corr_coef, corr_p = stats.pearsonr(data1, data2)
                corr_tests[f"{var1}_vs_{var2}"] = {
                    'correlation': corr_coef,
                    'p_value': corr_p,
                    'significant': corr_p < 0.05
                }

        statistical_results['correlations'] = corr_tests

        self.hasil_analisis['statistical'] = statistical_results

        self.update_statistical_display()
        self.plot_statistical()

    def analisis_montecarlo(self):
        """Monte Carlo simulation untuk analisis risiko"""
        self.update_progress(30, "Menjalankan simulasi Monte Carlo...")

        n_sims = self.sim_runs.get()

        # Estimate parameters from historical data
        inflasi_data = np.array(self.data_historis['inflasi'])
        suku_bunga_data = np.array(self.data_historis['suku_bunga'])
        usd_data = np.array(self.data_historis['indeks_usd'])

        # Parameter distributions (assume normal)
        inflasi_params = (np.mean(inflasi_data), np.std(inflasi_data))
        suku_bunga_params = (np.mean(suku_bunga_data), np.std(suku_bunga_data))
        usd_params = (np.mean(usd_data), np.std(usd_data))

        # Prepare model
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T
        y = np.array(self.data_historis['harga_emas'])

        # Monte Carlo simulation
        predictions = []

        for i in range(n_sims):
            # Generate random parameters
            inflasi_sim = np.random.normal(inflasi_params[0], inflasi_params[1])
            suku_bunga_sim = np.random.normal(suku_bunga_params[0], suku_bunga_params[1])
            usd_sim = np.random.normal(usd_params[0], usd_params[1])

            # Make prediction
            params = [inflasi_sim, suku_bunga_sim, usd_sim]
            try:
                pred = self.predict_with_params(M, y, params)
                predictions.append(pred)
            except:
                continue

            if i % 100 == 0:
                self.update_progress(30 + (i / n_sims) * 60, f"Simulasi {i}/{n_sims}")

        predictions = np.array(predictions)

        # Calculate statistics
        confidence = self.confidence_level.get()
        percentiles = [
            (1 - confidence) / 2 * 100,
            50,
            (1 - (1 - confidence) / 2) * 100
        ]

        ci_lower, median, ci_upper = np.percentile(predictions, percentiles)

        montecarlo_results = {
            'predictions': predictions,
            'statistics': {
                'mean': np.mean(predictions),
                'median': median,
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'var': np.var(predictions)
            },
            'risk_metrics': {
                'var_95': np.percentile(predictions, 5),
                'cvar_95': np.mean(predictions[predictions <= np.percentile(predictions, 5)]),
                'probability_loss': np.mean(predictions < np.mean(self.data_historis['harga_emas'])) * 100
            }
        }

        self.hasil_analisis['montecarlo'] = montecarlo_results

        self.update_montecarlo_display()
        self.plot_montecarlo()

    def analisis_features(self):
        """Analisis feature engineering dan importance"""
        self.update_progress(30, "Menganalisis features...")

        # Original features
        original_features = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T

        # Create engineered features
        inflasi = np.array(self.data_historis['inflasi'])
        suku_bunga = np.array(self.data_historis['suku_bunga'])
        usd = np.array(self.data_historis['indeks_usd'])
        tahun = np.array(self.data_historis['tahun'])

        # Feature engineering
        engineered_features = []
        feature_names = []

        # Original features
        engineered_features.extend([inflasi, suku_bunga, usd])
        feature_names.extend(['Inflasi', 'Suku_Bunga', 'USD_Index'])

        # Interaction features
        engineered_features.append(inflasi * suku_bunga)
        feature_names.append('Inflasi_x_Suku_Bunga')

        engineered_features.append(inflasi * usd)
        feature_names.append('Inflasi_x_USD')

        engineered_features.append(suku_bunga * usd)
        feature_names.append('Suku_Bunga_x_USD')

        # Polynomial features
        engineered_features.append(inflasi ** 2)
        feature_names.append('Inflasi_Squared')

        engineered_features.append(suku_bunga ** 2)
        feature_names.append('Suku_Bunga_Squared')

        # Ratio features
        if np.all(suku_bunga != 0):
            engineered_features.append(inflasi / suku_bunga)
            feature_names.append('Inflasi_per_Suku_Bunga')

        # Time-based features
        years_normalized = (tahun - np.min(tahun)) / (np.max(tahun) - np.min(tahun))
        engineered_features.append(years_normalized)
        feature_names.append('Year_Normalized')

        # Combine all features
        X_engineered = np.column_stack(engineered_features)
        y = np.array(self.data_historis['harga_emas'])

        # Feature importance using different methods
        from sklearn.feature_selection import mutual_info_regression, f_regression

        # Correlation-based importance
        corr_importance = []
        for i in range(X_engineered.shape[1]):
            corr = np.corrcoef(X_engineered[:, i], y)[0, 1]
            corr_importance.append(abs(corr))

        # Mutual information
        mi_scores = mutual_info_regression(X_engineered, y)

        # F-statistics
        f_scores, f_p_values = f_regression(X_engineered, y)

        # Random Forest importance
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_engineered, y)
        rf_importance = rf_model.feature_importances_

        feature_analysis = {
            'feature_names': feature_names,
            'correlation_importance': corr_importance,
            'mutual_info_scores': mi_scores,
            'f_scores': f_scores,
            'f_p_values': f_p_values,
            'rf_importance': rf_importance,
            'feature_matrix': X_engineered
        }

        self.hasil_analisis['features'] = feature_analysis

        self.update_features_display()
        self.plot_features()

    def predict_with_params(self, M, y, params):
        """Helper function untuk prediksi dengan parameter tertentu"""
        # SVD prediction
        U, S, VT = np.linalg.svd(M, full_matrices=False)
        V = VT.T
        UT_y = U.T @ y
        S_inv = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > 1e-10:
                S_inv[i] = 1 / S[i]
        omega = V @ (S_inv * UT_y)
        return np.array(params) @ omega

    # Update display methods for each analysis type
    def update_korelasi_display(self):
        """Update correlation analysis display"""
        if self.hasil_analisis['korelasi'] is None:
            return

        result = self.hasil_analisis['korelasi']
        corr_matrix = result['matrix']
        p_values = result['p_values']
        variables = result['variables']

        output = "=== ANALISIS KORELASI ===\n\n"
        output += "Matriks Korelasi:\n"
        output += f"{'':12}"
        for var in variables:
            output += f"{var[:10]:>10}"
        output += "\n"

        for i, var1 in enumerate(variables):
            output += f"{var1[:10]:10}  "
            for j in range(len(variables)):
                output += f"{corr_matrix[i, j]:>9.3f}"
            output += "\n"

        output += "\nSignifikansi (p-values):\n"
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                significance = "***" if p_values[i, j] < 0.001 else "**" if p_values[i, j] < 0.01 else "*" if p_values[
                                                                                                                  i, j] < 0.05 else ""
                output += f"{variables[i]} vs {variables[j]}: r={corr_matrix[i, j]:.3f}, p={p_values[i, j]:.4f}{significance}\n"

        output += "\nInterpretasi:\n"
        output += "*** p < 0.001 (sangat signifikan)\n"
        output += "**  p < 0.01  (signifikan)\n"
        output += "*   p < 0.05  (agak signifikan)\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_sensitivitas_display(self):
        """Update sensitivity analysis display"""
        if self.hasil_analisis['sensitivitas'] is None:
            return

        result = self.hasil_analisis['sensitivitas']
        base_pred = result['base_prediction']
        sens_results = result['results']

        output = "=== ANALISIS SENSITIVITAS ===\n\n"
        output += f"Prediksi Base: ${base_pred:.2f}\n\n"

        for param_name, data in sens_results.items():
            output += f"{param_name.upper()}:\n"
            output += f"  Base Value: {data['base_value']:.3f}\n"
            output += "  Variasi -> Perubahan Prediksi:\n"

            for var, sens in zip(data['variations'], data['sensitivity']):
                output += f"    {var:+3d}% -> {sens:+6.2f}%\n"

            max_sens = max(abs(s) for s in data['sensitivity'])
            output += f"  Max Sensitivity: ±{max_sens:.2f}%\n\n"

        # Sensitivity ranking
        max_sensitivities = {}
        for param_name, data in sens_results.items():
            max_sensitivities[param_name] = max(abs(s) for s in data['sensitivity'])

        sorted_sens = sorted(max_sensitivities.items(), key=lambda x: x[1], reverse=True)
        output += "Ranking Sensitivitas (paling sensitif):\n"
        for i, (param, sens) in enumerate(sorted_sens, 1):
            output += f"{i}. {param}: ±{sens:.2f}%\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_trend_display(self):
        """Update trend analysis display"""
        if self.hasil_analisis['trend'] is None:
            return

        results = self.hasil_analisis['trend']

        output = "=== ANALISIS TREND ===\n\n"

        for var, data in results.items():
            output += f"{var.upper()}:\n"
            output += f"  Trend: {data['slope']:+.4f} per tahun\n"
            output += f"  R²: {data['r_squared']:.4f}\n"
            output += f"  Signifikansi: {'Signifikan' if data['p_value'] < 0.05 else 'Tidak Signifikan'} (p={data['p_value']:.4f})\n"
            output += f"  Rata-rata Growth Rate: {data['avg_growth']:+.2f}% per tahun\n"
            output += f"  Volatilitas: {data['volatility']:.4f}\n"

            if data['slope'] > 0:
                trend_desc = "Naik"
            elif data['slope'] < 0:
                trend_desc = "Turun"
            else:
                trend_desc = "Stabil"
            output += f"  Interpretasi: Trend {trend_desc}\n\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_outlier_display(self):
        """Update outlier analysis display"""
        if self.hasil_analisis['outlier'] is None:
            return

        results = self.hasil_analisis['outlier']
        tahun = self.data_historis['tahun']

        output = "=== ANALISIS OUTLIER ===\n\n"

        for var, data in results.items():
            output += f"{var.upper()}:\n"

            # Z-score outliers
            if len(data['z_outliers']) > 0:
                output += f"  Z-Score Outliers: {len(data['z_outliers'])} data point(s)\n"
                for idx in data['z_outliers']:
                    output += f"    Tahun {tahun[idx]}: Z-score = {data['z_scores'][idx]:.2f}\n"
            else:
                output += "  Z-Score Outliers: Tidak ada\n"

            # IQR outliers
            if len(data['iqr_outliers']) > 0:
                output += f"  IQR Outliers: {len(data['iqr_outliers'])} data point(s)\n"
                for idx in data['iqr_outliers']:
                    value = self.data_historis[var][idx]
                    output += f"    Tahun {tahun[idx]}: {value:.2f}\n"
            else:
                output += "  IQR Outliers: Tidak ada\n"

            # Statistics
            stats_data = data['statistics']
            output += f"  Mean: {stats_data['mean']:.3f}\n"
            output += f"  Median: {stats_data['median']:.3f}\n"
            output += f"  Std: {stats_data['std']:.3f}\n\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_timeseries_display(self):
        """Update time series analysis display"""
        if self.hasil_analisis['timeseries'] is None:
            return

        result = self.hasil_analisis['timeseries']

        output = "=== ANALISIS TIME SERIES ===\n\n"
        output += f"Slope Trend: {result['slope']:.4f} USD per tahun\n"
        output += f"Intercept: {result['intercept']:.2f} USD\n\n"

        # Stationarity analysis
        stats_data = result['stationarity']
        output += "Statistik Stationaritas:\n"
        output += f"  Mean: {stats_data['mean']:.2f}\n"
        output += f"  Std: {stats_data['std']:.2f}\n"
        output += f"  Trend Strength: {stats_data['trend_strength']:.4f}\n"
        output += f"  Residual Std: {stats_data['residual_std']:.2f}\n\n"

        # Decomposition summary
        output += "Komponen Dekomposisi:\n"
        output += f"  Trend: Linear dengan slope {result['slope']:.4f}\n"
        output += f"  Seasonal: Simple moving average\n"
        output += f"  Residual: Sisa setelah dikurangi trend dan seasonal\n\n"

        # Interpretation
        if abs(result['slope']) > 50:
            trend_strength = "Kuat"
        elif abs(result['slope']) > 20:
            trend_strength = "Sedang"
        else:
            trend_strength = "Lemah"

        trend_direction = "Naik" if result['slope'] > 0 else "Turun" if result['slope'] < 0 else "Stabil"
        output += f"Interpretasi: Trend {trend_direction} dengan kekuatan {trend_strength}\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_statistical_display(self):
        """Update statistical analysis display"""
        if self.hasil_analisis['statistical'] is None:
            return

        results = self.hasil_analisis['statistical']

        output = "=== UJI STATISTIK ===\n\n"

        # Descriptive statistics for each variable
        for var in ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']:
            if var in results:
                data = results[var]
                desc = data['descriptive']
                norm = data['normality']
                ci = data['confidence_interval']

                output += f"{var.upper()}:\n"
                output += f"  Mean: {desc['mean']:.4f}\n"
                output += f"  Median: {desc['median']:.4f}\n"
                output += f"  Std: {desc['std']:.4f}\n"
                output += f"  Skewness: {desc['skewness']:.4f}\n"
                output += f"  Kurtosis: {desc['kurtosis']:.4f}\n"
                output += f"  Range: {desc['range']:.4f}\n"
                output += f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
                output += f"  Normalitas: {'Ya' if norm['is_normal'] else 'Tidak'} (p={norm['shapiro_p']:.4f})\n\n"

        # Correlation significance
        output += "SIGNIFIKANSI KORELASI:\n"
        if 'correlations' in results:
            for pair, data in results['correlations'].items():
                significance = "Signifikan" if data['significant'] else "Tidak Signifikan"
                output += f"  {pair.replace('_vs_', ' vs ')}: r={data['correlation']:.3f}, p={data['p_value']:.4f} ({significance})\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_montecarlo_display(self):
        """Update Monte Carlo analysis display"""
        if self.hasil_analisis['montecarlo'] is None:
            return

        result = self.hasil_analisis['montecarlo']
        stats_data = result['statistics']
        risk_data = result['risk_metrics']

        output = "=== SIMULASI MONTE CARLO ===\n\n"
        output += f"Jumlah Simulasi: {len(result['predictions'])}\n\n"

        output += "STATISTIK PREDIKSI:\n"
        output += f"  Mean: ${stats_data['mean']:.2f}\n"
        output += f"  Median: ${stats_data['median']:.2f}\n"
        output += f"  Std: ${stats_data['std']:.2f}\n"
        output += f"  Min: ${stats_data['min']:.2f}\n"
        output += f"  Max: ${stats_data['max']:.2f}\n"
        output += f"  95% CI: [${stats_data['ci_lower']:.2f}, ${stats_data['ci_upper']:.2f}]\n\n"

        output += "METRIK RISIKO:\n"
        output += f"  VaR (95%): ${risk_data['var_95']:.2f}\n"
        output += f"  CVaR (95%): ${risk_data['cvar_95']:.2f}\n"
        output += f"  Probability of Loss: {risk_data['probability_loss']:.1f}%\n\n"

        # Risk interpretation
        current_avg = np.mean(self.data_historis['harga_emas'])
        upside_potential = ((stats_data['mean'] - current_avg) / current_avg) * 100

        output += "INTERPRETASI:\n"
        output += f"  Expected Return: {upside_potential:+.1f}%\n"
        output += f"  Volatilitas: ${stats_data['std']:.2f} ({(stats_data['std'] / stats_data['mean']) * 100:.1f}%)\n"

        if risk_data['probability_loss'] < 25:
            risk_level = "Rendah"
        elif risk_data['probability_loss'] < 50:
            risk_level = "Sedang"
        else:
            risk_level = "Tinggi"

        output += f"  Tingkat Risiko: {risk_level}\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    def update_features_display(self):
        """Update feature analysis display"""
        if self.hasil_analisis['features'] is None:
            return

        result = self.hasil_analisis['features']
        feature_names = result['feature_names']

        output = "=== ANALISIS FEATURE ===\n\n"
        output += f"Total Features: {len(feature_names)}\n\n"

        # Feature importance ranking
        importance_methods = [
            ('Correlation', result['correlation_importance']),
            ('Mutual Info', result['mutual_info_scores']),
            ('Random Forest', result['rf_importance'])
        ]

        for method_name, scores in importance_methods:
            output += f"{method_name.upper()} IMPORTANCE:\n"

            # Sort by importance
            importance_pairs = list(zip(feature_names, scores))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, score) in enumerate(importance_pairs[:8], 1):  # Top 8
                output += f"  {i:2d}. {feature:20s}: {score:.4f}\n"
            output += "\n"

        # F-statistics
        output += "F-STATISTICS (Significance):\n"
        f_pairs = list(zip(feature_names, result['f_scores'], result['f_p_values']))
        f_pairs.sort(key=lambda x: x[1], reverse=True)

        for feature, f_score, p_value in f_pairs[:8]:
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            output += f"  {feature:20s}: F={f_score:8.2f}, p={p_value:.4f}{significance}\n"

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, output)
        self.teks_analisis.config(state=tk.DISABLED)

    # Plotting methods for each analysis type
    def plot_korelasi(self):
        """Plot correlation heatmap"""
        if self.hasil_analisis['korelasi'] is None:
            return

        self.gambar_analisis.clear()
        ax = self.gambar_analisis.add_subplot(111)

        result = self.hasil_analisis['korelasi']
        corr_matrix = result['matrix']
        variables = result['variables']

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(variables)))
        ax.set_yticks(range(len(variables)))
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.set_yticklabels(variables)

        # Add correlation values as text
        for i in range(len(variables)):
            for j in range(len(variables)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

        # Add colorbar
        cbar = self.gambar_analisis.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')

        ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_sensitivitas(self):
        """Plot sensitivity analysis"""
        if self.hasil_analisis['sensitivitas'] is None:
            return

        self.gambar_analisis.clear()

        result = self.hasil_analisis['sensitivitas']
        sens_results = result['results']

        # Create subplots for each parameter
        n_params = len(sens_results)
        if n_params <= 2:
            rows, cols = 1, n_params
        else:
            rows, cols = 2, 2

        param_names = list(sens_results.keys())
        colors = ['blue', 'red', 'green']

        for i, param_name in enumerate(param_names):
            ax = self.gambar_analisis.add_subplot(rows, cols, i + 1)

            data = sens_results[param_name]
            variations = data['variations']
            sensitivity = data['sensitivity']

            ax.plot(variations, sensitivity, 'o-', color=colors[i % len(colors)],
                    linewidth=2, markersize=6)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

            ax.set_xlabel('Parameter Variation (%)')
            ax.set_ylabel('Prediction Change (%)')
            ax.set_title(f'Sensitivity: {param_name.title()}')
            ax.grid(True, alpha=0.3)

        self.gambar_analisis.suptitle('Sensitivity Analysis', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_trend(self):
        """Plot trend analysis"""
        if self.hasil_analisis['trend'] is None:
            return

        self.gambar_analisis.clear()

        results = self.hasil_analisis['trend']
        tahun = self.data_historis['tahun']

        # Create 2x2 subplot
        variables = ['harga_emas', 'inflasi', 'suku_bunga', 'indeks_usd']
        titles = ['Harga Emas (USD)', 'Inflasi (%)', 'Suku Bunga (%)', 'Indeks USD']
        colors = ['gold', 'red', 'blue', 'green']

        for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
            ax = self.gambar_analisis.add_subplot(2, 2, i + 1)

            data = np.array(self.data_historis[var])
            trend_data = results[var]

            # Plot actual data
            ax.scatter(tahun, data, color=color, alpha=0.7, s=50, label='Data')

            # Plot trend line
            trend_line = trend_data['slope'] * np.arange(len(tahun)) + trend_data['intercept']
            ax.plot(tahun, trend_line, color='black', linestyle='--', linewidth=2, label='Trend')

            # Plot moving average if available
            if len(trend_data['moving_avg']) > 0:
                ma_years = tahun[1:-1] if len(trend_data['moving_avg']) == len(tahun) - 2 else tahun[:len(
                    trend_data['moving_avg'])]
                ax.plot(ma_years, trend_data['moving_avg'], color=color, alpha=0.5, linewidth=2, label='Moving Avg')

            ax.set_title(f'{title}\nTrend: {trend_data["slope"]:+.3f}/year (R²={trend_data["r_squared"]:.3f})')
            ax.set_xlabel('Tahun')
            ax.set_ylabel(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        self.gambar_analisis.suptitle('Trend Analysis', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_outlier(self):
        """Plot outlier analysis"""
        if self.hasil_analisis['outlier'] is None:
            return

        self.gambar_analisis.clear()

        results = self.hasil_analisis['outlier']
        tahun = self.data_historis['tahun']

        variables = ['harga_emas', 'inflasi', 'suku_bunga', 'indeks_usd']
        titles = ['Harga Emas (USD)', 'Inflasi (%)', 'Suku Bunga (%)', 'Indeks USD']
        colors = ['gold', 'red', 'blue', 'green']

        for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
            ax = self.gambar_analisis.add_subplot(2, 2, i + 1)

            data = np.array(self.data_historis[var])
            outlier_data = results[var]

            # Plot all data
            ax.scatter(tahun, data, color=color, alpha=0.6, s=50, label='Normal')

            # Highlight outliers
            z_outliers = outlier_data['z_outliers']
            iqr_outliers = outlier_data['iqr_outliers']

            if len(z_outliers) > 0:
                ax.scatter(np.array(tahun)[z_outliers], data[z_outliers],
                           color='red', s=100, marker='x', linewidth=3, label='Z-Score Outlier')

            if len(iqr_outliers) > 0:
                ax.scatter(np.array(tahun)[iqr_outliers], data[iqr_outliers],
                           color='orange', s=100, marker='^', label='IQR Outlier')

            # Add bounds for IQR method
            bounds = outlier_data['bounds']
            ax.axhline(y=bounds['upper'], color='orange', linestyle=':', alpha=0.7, label='IQR Bounds')
            ax.axhline(y=bounds['lower'], color='orange', linestyle=':', alpha=0.7)

            ax.set_title(f'{title}')
            ax.set_xlabel('Tahun')
            ax.set_ylabel(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        self.gambar_analisis.suptitle('Outlier Detection', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_timeseries(self):
        """Plot time series decomposition"""
        if self.hasil_analisis['timeseries'] is None:
            return

        self.gambar_analisis.clear()

        result = self.hasil_analisis['timeseries']
        tahun = self.data_historis['tahun']

        # Create 4 subplots for decomposition
        ax1 = self.gambar_analisis.add_subplot(4, 1, 1)
        ax2 = self.gambar_analisis.add_subplot(4, 1, 2)
        ax3 = self.gambar_analisis.add_subplot(4, 1, 3)
        ax4 = self.gambar_analisis.add_subplot(4, 1, 4)

        # Original series
        ax1.plot(tahun, result['original'], 'o-', color='blue', linewidth=2)
        ax1.set_title('Original Series')
        ax1.set_ylabel('Harga Emas (USD)')
        ax1.grid(True, alpha=0.3)

        # Trend
        ax2.plot(tahun, result['trend'], '-', color='red', linewidth=2)
        ax2.set_title('Trend Component')
        ax2.set_ylabel('Trend')
        ax2.grid(True, alpha=0.3)

        # Seasonal (if any pattern)
        ax3.plot(tahun, result['seasonal'], '-', color='green', linewidth=2)
        ax3.set_title('Seasonal Component')
        ax3.set_ylabel('Seasonal')
        ax3.grid(True, alpha=0.3)

        # Residual
        ax4.plot(tahun, result['residual'], 'o-', color='purple', linewidth=1, markersize=4)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Residual Component')
        ax4.set_ylabel('Residual')
        ax4.set_xlabel('Tahun')
        ax4.grid(True, alpha=0.3)

        self.gambar_analisis.suptitle('Time Series Decomposition', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_statistical(self):
        """Plot statistical analysis"""
        if self.hasil_analisis['statistical'] is None:
            return

        self.gambar_analisis.clear()

        results = self.hasil_analisis['statistical']
        variables = ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']

        # Create 2x2 subplot for normality plots
        for i, var in enumerate(variables):
            ax = self.gambar_analisis.add_subplot(2, 2, i + 1)

            data = np.array(self.data_historis[var])

            # Q-Q plot for normality
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f'{var.title()} - Q-Q Plot')
            ax.grid(True, alpha=0.3)

            # Add normality test result
            is_normal = results[var]['normality']['is_normal']
            p_value = results[var]['normality']['shapiro_p']
            color = 'green' if is_normal else 'red'
            status = 'Normal' if is_normal else 'Not Normal'

            ax.text(0.05, 0.95, f'{status}\np={p_value:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

        self.gambar_analisis.suptitle('Normality Testing (Q-Q Plots)', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_montecarlo(self):
        """Plot Monte Carlo simulation results"""
        if self.hasil_analisis['montecarlo'] is None:
            return

        self.gambar_analisis.clear()

        result = self.hasil_analisis['montecarlo']
        predictions = result['predictions']
        stats_data = result['statistics']

        # Create 2x2 subplot
        ax1 = self.gambar_analisis.add_subplot(2, 2, 1)
        ax2 = self.gambar_analisis.add_subplot(2, 2, 2)
        ax3 = self.gambar_analisis.add_subplot(2, 2, 3)
        ax4 = self.gambar_analisis.add_subplot(2, 2, 4)

        # 1. Histogram of predictions
        ax1.hist(predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(stats_data['mean'], color='red', linestyle='--', linewidth=2,
                    label=f'Mean: ${stats_data["mean"]:.0f}')
        ax1.axvline(stats_data['median'], color='green', linestyle='--', linewidth=2,
                    label=f'Median: ${stats_data["median"]:.0f}')
        ax1.axvline(stats_data['ci_lower'], color='orange', linestyle=':', linewidth=2, label=f'95% CI')
        ax1.axvline(stats_data['ci_upper'], color='orange', linestyle=':', linewidth=2)
        ax1.set_title('Distribution of Predictions')
        ax1.set_xlabel('Predicted Price (USD)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot
        ax2.boxplot(predictions)
        ax2.set_title('Box Plot of Predictions')
        ax2.set_ylabel('Predicted Price (USD)')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative distribution
        sorted_preds = np.sort(predictions)
        cumulative = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
        ax3.plot(sorted_preds, cumulative, linewidth=2)
        ax3.axvline(stats_data['ci_lower'], color='red', linestyle='--', alpha=0.7)
        ax3.axvline(stats_data['ci_upper'], color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Cumulative Distribution')
        ax3.set_xlabel('Predicted Price (USD)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)

        # 4. Risk metrics visualization
        risk_data = result['risk_metrics']
        metrics = ['VaR 95%', 'CVaR 95%', 'Mean', 'Median']
        values = [risk_data['var_95'], risk_data['cvar_95'], stats_data['mean'], stats_data['median']]
        colors = ['red', 'darkred', 'blue', 'green']

        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Risk Metrics')
        ax4.set_ylabel('Price (USD)')
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'${value:.0f}', ha='center', va='bottom')

        self.gambar_analisis.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    def plot_features(self):
        """Plot feature importance analysis"""
        if self.hasil_analisis['features'] is None:
            return

        self.gambar_analisis.clear()

        result = self.hasil_analisis['features']
        feature_names = result['feature_names']

        # Create 2x2 subplot for different importance measures
        ax1 = self.gambar_analisis.add_subplot(2, 2, 1)
        ax2 = self.gambar_analisis.add_subplot(2, 2, 2)
        ax3 = self.gambar_analisis.add_subplot(2, 2, 3)
        ax4 = self.gambar_analisis.add_subplot(2, 2, 4)

        # 1. Correlation importance
        corr_imp = result['correlation_importance']
        sorted_indices = np.argsort(corr_imp)[::-1][:8]  # Top 8

        ax1.barh(range(len(sorted_indices)), [corr_imp[i] for i in sorted_indices])
        ax1.set_yticks(range(len(sorted_indices)))
        ax1.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax1.set_title('Correlation Importance')
        ax1.set_xlabel('Absolute Correlation')

        # 2. Mutual Information
        mi_scores = result['mutual_info_scores']
        sorted_indices = np.argsort(mi_scores)[::-1][:8]

        ax2.barh(range(len(sorted_indices)), [mi_scores[i] for i in sorted_indices], color='orange')
        ax2.set_yticks(range(len(sorted_indices)))
        ax2.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax2.set_title('Mutual Information')
        ax2.set_xlabel('MI Score')

        # 3. Random Forest importance
        rf_imp = result['rf_importance']
        sorted_indices = np.argsort(rf_imp)[::-1][:8]

        ax3.barh(range(len(sorted_indices)), [rf_imp[i] for i in sorted_indices], color='green')
        ax3.set_yticks(range(len(sorted_indices)))
        ax3.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax3.set_title('Random Forest Importance')
        ax3.set_xlabel('Importance')

        # 4. F-statistics
        f_scores = result['f_scores']
        sorted_indices = np.argsort(f_scores)[::-1][:8]

        ax4.barh(range(len(sorted_indices)), [f_scores[i] for i in sorted_indices], color='red')
        ax4.set_yticks(range(len(sorted_indices)))
        ax4.set_yticklabels([feature_names[i] for i in sorted_indices])
        ax4.set_title('F-Statistics')
        ax4.set_xlabel('F-Score')

        self.gambar_analisis.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        self.gambar_analisis.tight_layout()
        self.kanvas_analisis.draw()

    # Additional utility methods
    def refresh_analisis_viz(self):
        """Refresh analysis visualization"""
        jenis = self.jenis_analisis.get()

        if jenis == "korelasi" and self.hasil_analisis['korelasi']:
            self.plot_korelasi()
        elif jenis == "sensitivitas" and self.hasil_analisis['sensitivitas']:
            self.plot_sensitivitas()
        elif jenis == "trend" and self.hasil_analisis['trend']:
            self.plot_trend()
        elif jenis == "outlier" and self.hasil_analisis['outlier']:
            self.plot_outlier()
        elif jenis == "timeseries" and self.hasil_analisis['timeseries']:
            self.plot_timeseries()
        elif jenis == "statistical" and self.hasil_analisis['statistical']:
            self.plot_statistical()
        elif jenis == "montecarlo" and self.hasil_analisis['montecarlo']:
            self.plot_montecarlo()
        elif jenis == "features" and self.hasil_analisis['features']:
            self.plot_features()

    def save_analisis_plot(self):
        """Save analysis plot"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filename:
                self.gambar_analisis.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Sukses", f"Plot berhasil disimpan: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan plot: {str(e)}")

    def export_analisis_hasil(self):
        """Export analysis results to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.teks_analisis.get(1.0, tk.END))
                messagebox.showinfo("Sukses", f"Hasil analisis berhasil diekspor: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekspor hasil: {str(e)}")

    def reset_analisis(self):
        """Reset analysis results"""
        self.hasil_analisis = {
            'korelasi': None,
            'sensitivitas': None,
            'trend': None,
            'outlier': None,
            'timeseries': None,
            'statistical': None,
            'montecarlo': None,
            'features': None
        }

        self.teks_analisis.config(state=tk.NORMAL)
        self.teks_analisis.delete(1.0, tk.END)
        self.teks_analisis.insert(tk.END, "Analisis telah direset. Pilih jenis analisis dan klik 'Jalankan Analisis'.")
        self.teks_analisis.config(state=tk.DISABLED)

        self.gambar_analisis.clear()
        self.kanvas_analisis.draw()

        self.reset_progress()

    def buat_tab_model_comparison(self):
        """Membuat tab perbandingan model yang komprehensif"""
        bingkai = ttk.Frame(self.tab_model_comparison)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Kontrol dan pengaturan
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Hasil dan visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri ===
        # Model Selection
        model_selection_frame = ttk.LabelFrame(frame_kiri, text="Pilih Model untuk Dibandingkan")
        model_selection_frame.pack(fill=tk.X, pady=5, padx=5)

        self.model_comparison_vars = {}
        models_info = [
            ("svd", "SVD + Least Squares", "Decomposition-based method"),
            ("linear", "Linear Regression", "Simple linear model"),
            ("ridge", "Ridge Regression", "L2 regularization"),
            ("lasso", "Lasso Regression", "L1 regularization"),
            ("random_forest", "Random Forest", "Ensemble method")
        ]

        for i, (key, name, desc) in enumerate(models_info):
            var = tk.BooleanVar(value=True)
            self.model_comparison_vars[key] = var

            cb = ttk.Checkbutton(model_selection_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

            # Tooltip dengan deskripsi
            self.buat_tooltip(cb, desc)

        # Validation Settings
        validation_frame = ttk.LabelFrame(frame_kiri, text="Pengaturan Validasi")
        validation_frame.pack(fill=tk.X, pady=5, padx=5)

        # Cross-validation folds
        ttk.Label(validation_frame, text="CV Folds:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.cv_folds = tk.IntVar(value=5)
        cv_spinbox = ttk.Spinbox(validation_frame, from_=3, to=10, textvariable=self.cv_folds, width=10)
        cv_spinbox.grid(row=0, column=1, padx=5, pady=5)

        # Test split ratio
        ttk.Label(validation_frame, text="Test Split:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.test_split = tk.DoubleVar(value=0.2)
        split_spinbox = ttk.Spinbox(validation_frame, from_=0.1, to=0.5, increment=0.1,
                                    textvariable=self.test_split, width=10)
        split_spinbox.grid(row=1, column=1, padx=5, pady=5)

        # Random state
        ttk.Label(validation_frame, text="Random State:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.random_state = tk.IntVar(value=42)
        state_spinbox = ttk.Spinbox(validation_frame, from_=1, to=100, textvariable=self.random_state, width=10)
        state_spinbox.grid(row=2, column=1, padx=5, pady=5)

        # Comparison Options
        options_frame = ttk.LabelFrame(frame_kiri, text="Opsi Perbandingan")
        options_frame.pack(fill=tk.X, pady=5, padx=5)

        self.include_feature_importance = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Analisis Feature Importance",
                        variable=self.include_feature_importance).pack(anchor=tk.W, padx=5, pady=2)

        self.include_residuals = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Analisis Residuals",
                        variable=self.include_residuals).pack(anchor=tk.W, padx=5, pady=2)

        self.include_statistical_tests = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Uji Statistik",
                        variable=self.include_statistical_tests).pack(anchor=tk.W, padx=5, pady=2)

        self.include_time_analysis = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Analisis Waktu Training",
                        variable=self.include_time_analysis).pack(anchor=tk.W, padx=5, pady=2)

        # Control Buttons
        control_frame = ttk.Frame(frame_kiri)
        control_frame.pack(fill=tk.X, pady=10, padx=5)

        ttk.Button(control_frame, text="🔄 Jalankan Perbandingan",
                   command=self.jalankan_model_comparison).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="📊 Export Hasil",
                   command=self.export_comparison_results).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🔄 Reset",
                   command=self.reset_model_comparison).pack(fill=tk.X, pady=2)

        # Results Summary
        summary_frame = ttk.LabelFrame(frame_kiri, text="Ringkasan Hasil")
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.summary_text = tk.Text(summary_frame, height=15, wrap=tk.WORD, font=("Consolas", 9))
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)

        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan ===
        # Visualization Controls
        viz_control_frame = ttk.LabelFrame(frame_kanan, text="Kontrol Visualisasi")
        viz_control_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(viz_control_frame, text="Jenis Visualisasi:").pack(side=tk.LEFT, padx=5)
        self.comparison_viz_type = tk.StringVar(value="metrics")
        viz_combo = ttk.Combobox(viz_control_frame, textvariable=self.comparison_viz_type,
                                 values=["metrics", "predictions", "residuals", "feature_importance",
                                         "cv_scores", "statistical"], width=15, state="readonly")
        viz_combo.pack(side=tk.LEFT, padx=5)
        viz_combo.bind("<<ComboboxSelected>>", self.update_comparison_visualization)

        ttk.Button(viz_control_frame, text="🔄 Refresh",
                   command=self.update_comparison_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_control_frame, text="💾 Save Plot",
                   command=self.save_comparison_plot).pack(side=tk.LEFT, padx=5)

        # Results Table
        table_frame = ttk.LabelFrame(frame_kanan, text="Tabel Perbandingan")
        table_frame.pack(fill=tk.X, pady=5, padx=5)

        # Treeview untuk hasil
        columns = ("Model", "R²", "RMSE", "MAE", "MAPE", "CV Mean", "CV Std", "Rank")
        self.comparison_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)

        # Setup columns
        for col in columns:
            self.comparison_tree.heading(col, text=col, command=lambda c=col: self.sort_comparison_results(c))
            if col == "Model":
                self.comparison_tree.column(col, width=120)
            else:
                self.comparison_tree.column(col, width=80, anchor=tk.CENTER)

        # Scrollbar untuk tabel
        table_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.comparison_tree.yview)
        self.comparison_tree.configure(yscrollcommand=table_scroll.set)

        self.comparison_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Visualization Area
        viz_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Perbandingan")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.gambar_comparison = Figure(figsize=(12, 8), dpi=100)
        self.kanvas_comparison = FigureCanvasTkAgg(self.gambar_comparison, viz_frame)

        # Toolbar navigasi
        toolbar_comparison_frame = ttk.Frame(viz_frame)
        toolbar_comparison_frame.pack(fill=tk.X, padx=5, pady=2)

        self.navbar_comparison = NavigationToolbar2Tk(self.kanvas_comparison, toolbar_comparison_frame)
        self.navbar_comparison.update()

        self.kanvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize comparison results storage
        self.comparison_results = {}

    def jalankan_model_comparison(self):
        """Menjalankan perbandingan model yang komprehensif"""
        try:
            self.update_progress(10, "Mempersiapkan data untuk perbandingan...")

            # Get selected models
            selected_models = [model for model, var in self.model_comparison_vars.items() if var.get()]

            if not selected_models:
                messagebox.showwarning("Warning", "Pilih minimal satu model untuk dibandingkan!")
                return

            # Prepare data
            M = np.array([
                self.data_historis['inflasi'],
                self.data_historis['suku_bunga'],
                self.data_historis['indeks_usd']
            ]).T
            y = np.array(self.data_historis['harga_emas'])

            # Split data for validation
            from sklearn.model_selection import train_test_split
            test_size = self.test_split.get()
            random_state = self.random_state.get()

            X_train, X_test, y_train, y_test = train_test_split(
                M, y, test_size=test_size, random_state=random_state
            )

            self.update_progress(30, "Menjalankan model...")

            # Initialize results
            results = {}

            for i, model_name in enumerate(selected_models):
                self.update_progress(30 + (i * 60 / len(selected_models)), f"Evaluasi model: {model_name}")

                try:
                    result = self.evaluate_single_model(model_name, X_train, X_test, y_train, y_test, M, y)
                    results[model_name] = result
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    continue

            self.comparison_results = results

            self.update_progress(95, "Memperbarui tampilan...")

            # Update displays
            self.update_comparison_table()
            self.update_comparison_summary()
            self.update_comparison_visualization()

            self.update_progress(100, "Perbandingan model selesai!")

            messagebox.showinfo("Sukses", f"Perbandingan {len(results)} model berhasil diselesaikan!")

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
            self.reset_progress()

    def evaluate_single_model(self, model_name, X_train, X_test, y_train, y_test, X_full, y_full):
        """Evaluasi satu model secara komprehensif"""
        import time
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        start_time = time.time()

        # Initialize model
        if model_name == "svd":
            # SVD doesn't use sklearn interface, handle separately
            model = None
            # SVD Training
            U, S, VT = np.linalg.svd(X_train, full_matrices=False)
            V = VT.T
            UT_y = U.T @ y_train
            S_inv = np.zeros_like(S)
            for i in range(len(S)):
                if S[i] > 1e-10:
                    S_inv[i] = 1 / S[i]
            omega = V @ (S_inv * UT_y)

            # Predictions
            y_train_pred = X_train @ omega
            y_test_pred = X_test @ omega
            y_full_pred = X_full @ omega

            # Feature importance (magnitude of omega coefficients)
            feature_importance = np.abs(omega)

        else:
            # ML Models
            if model_name == "linear":
                model = LinearRegression()
            elif model_name == "ridge":
                model = Ridge(alpha=1.0, random_state=self.random_state.get())
            elif model_name == "lasso":
                model = Lasso(alpha=1.0, random_state=self.random_state.get())
            elif model_name == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state.get())

            # Training
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_full_pred = model.predict(X_full)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_)
            else:
                feature_importance = np.ones(X_train.shape[1])

        training_time = time.time() - start_time

        # Calculate metrics
        # Training metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

        # Test metrics
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

        # Full dataset metrics
        full_r2 = r2_score(y_full, y_full_pred)
        full_rmse = np.sqrt(mean_squared_error(y_full, y_full_pred))
        full_mae = mean_absolute_error(y_full, y_full_pred)
        full_mape = np.mean(np.abs((y_full - y_full_pred) / y_full)) * 100

        # Cross-validation
        cv_scores = None
        cv_mean = 0
        cv_std = 0

        if model is not None:  # Skip CV for SVD
            try:
                cv_folds = self.cv_folds.get()
                cv_scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring='r2')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            except:
                pass

        # Residuals
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred
        residuals_full = y_full - y_full_pred

        # Additional statistics
        predictions_variance = np.var(y_full_pred)
        residuals_normality = None
        if len(residuals_full) > 3:
            try:
                from scipy.stats import shapiro
                _, residuals_normality = shapiro(residuals_full)
            except:
                pass

        return {
            'model': model,
            'model_name': model_name,
            'training_time': training_time,

            # Predictions
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'y_full_pred': y_full_pred,

            # Training metrics
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_mape': train_mape,

            # Test metrics
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,

            # Full dataset metrics
            'full_r2': full_r2,
            'full_rmse': full_rmse,
            'full_mae': full_mae,
            'full_mape': full_mape,

            # Cross-validation
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,

            # Feature analysis
            'feature_importance': feature_importance,

            # Residuals
            'residuals_train': residuals_train,
            'residuals_test': residuals_test,
            'residuals_full': residuals_full,

            # Additional stats
            'predictions_variance': predictions_variance,
            'residuals_normality_p': residuals_normality,

            # SVD specific (if applicable)
            'omega': omega if model_name == "svd" else None
        }

    def update_comparison_table(self):
        """Update comparison results table"""
        # Clear existing items
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)

        if not self.comparison_results:
            return

        # Calculate rankings based on test R²
        ranked_results = sorted(self.comparison_results.items(),
                                key=lambda x: x[1]['test_r2'], reverse=True)

        for rank, (model_name, result) in enumerate(ranked_results, 1):
            model_display = model_name.upper().replace('_', ' ')

            values = (
                model_display,
                f"{result['test_r2']:.4f}",
                f"{result['test_rmse']:.2f}",
                f"{result['test_mae']:.2f}",
                f"{result['test_mape']:.2f}%",
                f"{result['cv_mean']:.4f}" if result['cv_mean'] > 0 else "N/A",
                f"{result['cv_std']:.4f}" if result['cv_std'] > 0 else "N/A",
                str(rank)
            )

            # Color coding based on rank
            tags = []
            if rank == 1:
                tags = ['best']
            elif rank == len(ranked_results):
                tags = ['worst']

            item = self.comparison_tree.insert("", "end", values=values, tags=tags)

        # Configure tags for coloring
        self.comparison_tree.tag_configure('best', background='lightgreen')
        self.comparison_tree.tag_configure('worst', background='lightcoral')

    def update_comparison_summary(self):
        """Update comparison summary text"""
        if not self.comparison_results:
            return

        summary = "=== RINGKASAN PERBANDINGAN MODEL ===\n\n"

        # Overall ranking
        ranked_results = sorted(self.comparison_results.items(),
                                key=lambda x: x[1]['test_r2'], reverse=True)

        summary += "RANKING BERDASARKAN TEST R²:\n"
        for rank, (model_name, result) in enumerate(ranked_results, 1):
            summary += f"{rank}. {model_name.upper()}: R² = {result['test_r2']:.4f}\n"

        summary += "\n" + "=" * 50 + "\n\n"

        # Best performing model details
        best_model, best_result = ranked_results[0]
        summary += f"MODEL TERBAIK: {best_model.upper()}\n"
        summary += f"• Test R²: {best_result['test_r2']:.4f}\n"
        summary += f"• Test RMSE: {best_result['test_rmse']:.2f}\n"
        summary += f"• Test MAE: {best_result['test_mae']:.2f}\n"
        summary += f"• Test MAPE: {best_result['test_mape']:.2f}%\n"

        if best_result['cv_mean'] > 0:
            summary += f"• CV Mean R²: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}\n"

        if self.include_time_analysis.get():
            summary += f"• Training Time: {best_result['training_time']:.4f}s\n"

        summary += "\n" + "=" * 50 + "\n\n"

        # Detailed comparison
        summary += "PERBANDINGAN DETAIL:\n\n"

        metrics = ['test_r2', 'test_rmse', 'test_mae', 'test_mape']
        metric_names = ['R²', 'RMSE', 'MAE', 'MAPE']

        for metric, name in zip(metrics, metric_names):
            summary += f"{name}:\n"
            sorted_by_metric = sorted(self.comparison_results.items(),
                                      key=lambda x: x[1][metric],
                                      reverse=(metric == 'test_r2'))  # R² higher is better

            for rank, (model_name, result) in enumerate(sorted_by_metric, 1):
                value = result[metric]
                if metric == 'test_mape':
                    summary += f"  {rank}. {model_name.upper()}: {value:.2f}%\n"
                else:
                    summary += f"  {rank}. {model_name.upper()}: {value:.4f}\n"
            summary += "\n"

        # Statistical significance (if enough models)
        if len(self.comparison_results) >= 2 and self.include_statistical_tests.get():
            summary += "UJI STATISTIK:\n"
            # Simple comparison of best vs others
            best_cv = best_result.get('cv_scores')
            if best_cv is not None:
                for model_name, result in self.comparison_results.items():
                    if model_name != best_model and result.get('cv_scores') is not None:
                        try:
                            from scipy.stats import ttest_rel
                            _, p_value = ttest_rel(best_cv, result['cv_scores'])
                            significance = "Signifikan" if p_value < 0.05 else "Tidak Signifikan"
                            summary += f"  {best_model.upper()} vs {model_name.upper()}: p={p_value:.4f} ({significance})\n"
                        except:
                            pass
            summary += "\n"

        # Recommendations
        summary += "REKOMENDASI:\n"

        # Performance-based recommendation
        if best_result['test_r2'] > 0.8:
            performance_level = "Sangat Baik"
        elif best_result['test_r2'] > 0.6:
            performance_level = "Baik"
        elif best_result['test_r2'] > 0.4:
            performance_level = "Cukup"
        else:
            performance_level = "Perlu Perbaikan"

        summary += f"• Model terbaik ({best_model.upper()}) memiliki performa: {performance_level}\n"

        # Check for overfitting
        if abs(best_result['train_r2'] - best_result['test_r2']) > 0.1:
            summary += f"• Perhatian: Kemungkinan overfitting pada {best_model.upper()}\n"

        # Feature importance insight
        if self.include_feature_importance.get() and 'feature_importance' in best_result:
            features = ['Inflasi', 'Suku Bunga', 'Indeks USD']
            importance = best_result['feature_importance']
            most_important = features[np.argmax(importance)]
            summary += f"• Fitur paling penting untuk {best_model.upper()}: {most_important}\n"

        summary += f"\nDiupdate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Update text widget
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)

    def update_comparison_visualization(self, event=None):
        """Update comparison visualization based on selected type"""
        if not self.comparison_results:
            return

        viz_type = self.comparison_viz_type.get()

        self.gambar_comparison.clear()

        if viz_type == "metrics":
            self.plot_metrics_comparison()
        elif viz_type == "predictions":
            self.plot_predictions_comparison()
        elif viz_type == "residuals":
            self.plot_residuals_comparison()
        elif viz_type == "feature_importance":
            self.plot_feature_importance_comparison()
        elif viz_type == "cv_scores":
            self.plot_cv_scores_comparison()
        elif viz_type == "statistical":
            self.plot_statistical_comparison()

        self.gambar_comparison.tight_layout()
        self.kanvas_comparison.draw()

    def plot_metrics_comparison(self):
        """Plot metrics comparison bar charts"""
        # Create 2x2 subplot for different metrics
        ax1 = self.gambar_comparison.add_subplot(221)
        ax2 = self.gambar_comparison.add_subplot(222)
        ax3 = self.gambar_comparison.add_subplot(223)
        ax4 = self.gambar_comparison.add_subplot(224)

        models = list(self.comparison_results.keys())
        model_names = [m.upper().replace('_', ' ') for m in models]
        colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(models)]

        # R² Score
        r2_scores = [self.comparison_results[m]['test_r2'] for m in models]
        bars1 = ax1.bar(model_names, r2_scores, color=colors, alpha=0.7)
        ax1.set_title('R² Score (Test)')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontsize=9)

        # RMSE
        rmse_scores = [self.comparison_results[m]['test_rmse'] for m in models]
        bars2 = ax2.bar(model_names, rmse_scores, color=colors, alpha=0.7)
        ax2.set_title('RMSE (Test)')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=9)

        # MAE
        mae_scores = [self.comparison_results[m]['test_mae'] for m in models]
        bars3 = ax3.bar(model_names, mae_scores, color=colors, alpha=0.7)
        ax3.set_title('MAE (Test)')
        ax3.set_ylabel('MAE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        for bar, score in zip(bars3, mae_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=9)

        # MAPE
        mape_scores = [self.comparison_results[m]['test_mape'] for m in models]
        bars4 = ax4.bar(model_names, mape_scores, color=colors, alpha=0.7)
        ax4.set_title('MAPE (Test)')
        ax4.set_ylabel('MAPE (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        for bar, score in zip(bars4, mape_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{score:.1f}%', ha='center', va='bottom', fontsize=9)

    def plot_predictions_comparison(self):
        """Plot actual vs predicted comparison"""
        y_full = np.array(self.data_historis['harga_emas'])

        n_models = len(self.comparison_results)
        if n_models <= 2:
            rows, cols = 1, n_models
        elif n_models <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (model_name, result) in enumerate(self.comparison_results.items()):
            ax = self.gambar_comparison.add_subplot(rows, cols, i + 1)

            y_pred = result['y_full_pred']

            # Scatter plot
            ax.scatter(y_full, y_pred, color=colors[i % len(colors)], alpha=0.7, s=50)

            # Perfect prediction line
            min_val = min(np.min(y_full), np.min(y_pred))
            max_val = max(np.max(y_full), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)

            # R² annotation
            ax.text(0.05, 0.95, f'R² = {result["full_r2"]:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name.upper()}')
            ax.grid(True, alpha=0.3)

    def plot_residuals_comparison(self):
        """Plot residuals comparison"""
        n_models = len(self.comparison_results)
        if n_models <= 2:
            rows, cols = 1, n_models
        else:
            rows, cols = 2, (n_models + 1) // 2

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (model_name, result) in enumerate(self.comparison_results.items()):
            ax = self.gambar_comparison.add_subplot(rows, cols, i + 1)

            residuals = result['residuals_full']
            y_pred = result['y_full_pred']

            # Residuals vs fitted
            ax.scatter(y_pred, residuals, color=colors[i % len(colors)], alpha=0.7, s=50)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)

            # Add statistics
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{model_name.upper()} Residuals')
            ax.grid(True, alpha=0.3)

    def plot_feature_importance_comparison(self):
        """Plot feature importance comparison"""
        if not self.include_feature_importance.get():
            ax = self.gambar_comparison.add_subplot(111)
            ax.text(0.5, 0.5, 'Feature Importance Analysis\nnot enabled',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return

        features = ['Inflasi', 'Suku Bunga', 'Indeks USD']
        x = np.arange(len(features))
        width = 0.15

        ax = self.gambar_comparison.add_subplot(111)

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (model_name, result) in enumerate(self.comparison_results.items()):
            if 'feature_importance' in result:
                importance = result['feature_importance']
                # Normalize importance to 0-1 scale
                importance_norm = importance / np.max(importance)

                offset = (i - len(self.comparison_results) / 2) * width
                bars = ax.bar(x + offset, importance_norm, width,
                              label=model_name.upper(), color=colors[i % len(colors)], alpha=0.7)

                # Add value labels
                for bar, imp in zip(bars, importance_norm):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{imp:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Features')
        ax.set_ylabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_cv_scores_comparison(self):
        """Plot cross-validation scores comparison"""
        # Filter models that have CV scores
        cv_results = {k: v for k, v in self.comparison_results.items()
                      if v.get('cv_scores') is not None}

        if not cv_results:
            ax = self.gambar_comparison.add_subplot(111)
            ax.text(0.5, 0.5, 'Cross-Validation scores\nnot available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return

        ax = self.gambar_comparison.add_subplot(111)

        # Box plot of CV scores
        cv_data = []
        labels = []

        for model_name, result in cv_results.items():
            cv_data.append(result['cv_scores'])
            labels.append(model_name.upper())

        bp = ax.boxplot(cv_data, labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)

        ax.set_title('Cross-Validation Scores Distribution')
        ax.set_ylabel('R² Score')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    def plot_statistical_comparison(self):
        """Plot statistical analysis comparison"""
        # Create multiple subplots for different statistical views
        ax1 = self.gambar_comparison.add_subplot(221)
        ax2 = self.gambar_comparison.add_subplot(222)
        ax3 = self.gambar_comparison.add_subplot(223)
        ax4 = self.gambar_comparison.add_subplot(224)

        models = list(self.comparison_results.keys())

        # 1. Training vs Test Performance
        train_r2 = [self.comparison_results[m]['train_r2'] for m in models]
        test_r2 = [self.comparison_results[m]['test_r2'] for m in models]

        ax1.scatter(train_r2, test_r2, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax1.set_xlabel('Training R²')
        ax1.set_ylabel('Test R²')
        ax1.set_title('Training vs Test Performance')
        ax1.grid(True, alpha=0.3)

        # Add model labels
        for i, model in enumerate(models):
            ax1.annotate(model.upper()[:3], (train_r2[i], test_r2[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 2. Bias-Variance Analysis (simplified)
        model_names = [m.upper().replace('_', ' ') for m in models]
        bias_proxy = [self.comparison_results[m]['test_mae'] for m in models]  # MAE as bias proxy
        variance_proxy = [self.comparison_results[m]['cv_std'] if self.comparison_results[m]['cv_std'] > 0
                          else np.std(self.comparison_results[m]['residuals_full']) for m in models]

        ax2.scatter(bias_proxy, variance_proxy, s=100, alpha=0.7, c=range(len(models)), cmap='plasma')
        ax2.set_xlabel('Bias (MAE)')
        ax2.set_ylabel('Variance (CV Std/Residual Std)')
        ax2.set_title('Bias-Variance Trade-off')
        ax2.grid(True, alpha=0.3)

        for i, model in enumerate(model_names):
            ax2.annotate(model[:6], (bias_proxy[i], variance_proxy[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 3. Performance Distribution
        all_metrics = []
        metric_names = []
        model_labels = []

        for model in models:
            result = self.comparison_results[model]
            metrics = [result['test_r2'], result['test_rmse'] / 1000, result['test_mae'] / 1000]  # Normalize
            all_metrics.extend(metrics)
            metric_names.extend(['R²', 'RMSE/1000', 'MAE/1000'])
            model_labels.extend([model.upper()] * 3)

        # This would ideally be a more sophisticated plot
        ax3.bar(range(len(all_metrics)), all_metrics, alpha=0.7)
        ax3.set_title('Normalized Metrics Distribution')
        ax3.set_ylabel('Normalized Value')
        ax3.tick_params(axis='x', rotation=90)

        # 4. Model Complexity vs Performance
        # Simplified complexity measure
        complexity = []
        for model in models:
            if model == 'linear':
                complexity.append(1)
            elif model in ['ridge', 'lasso']:
                complexity.append(2)
            elif model == 'svd':
                complexity.append(3)
            elif model == 'random_forest':
                complexity.append(5)
            else:
                complexity.append(3)

        performance = [self.comparison_results[m]['test_r2'] for m in models]

        ax4.scatter(complexity, performance, s=100, alpha=0.7, c=range(len(models)), cmap='coolwarm')
        ax4.set_xlabel('Model Complexity (Relative)')
        ax4.set_ylabel('Test R²')
        ax4.set_title('Complexity vs Performance')
        ax4.grid(True, alpha=0.3)

        for i, model in enumerate(models):
            ax4.annotate(model.upper()[:5], (complexity[i], performance[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Additional utility methods for the comparison tab

    def sort_comparison_results(self, column):
        """Sort comparison results by column"""
        # Get current items
        items = [(self.comparison_tree.set(child, column), child)
                 for child in self.comparison_tree.get_children('')]

        # Sort items
        if column in ['R²', 'CV Mean']:
            # Higher is better
            items.sort(key=lambda x: float(x[0].replace('N/A', '0')), reverse=True)
        elif column in ['RMSE', 'MAE', 'MAPE', 'CV Std']:
            # Lower is better
            items.sort(key=lambda x: float(x[0].replace('%', '').replace('N/A', '999')))
        elif column == 'Rank':
            items.sort(key=lambda x: int(x[0]))
        else:
            items.sort(key=lambda x: x[0])

        # Reorder items
        for index, (val, child) in enumerate(items):
            self.comparison_tree.move(child, '', index)

    def export_comparison_results(self):
        """Export comparison results to file"""
        if not self.comparison_results:
            messagebox.showwarning("Warning", "Tidak ada hasil perbandingan untuk diekspor!")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Text files", "*.txt")]
            )

            if not filename:
                return

            if filename.endswith('.csv'):
                self.export_to_csv(filename)
            elif filename.endswith('.xlsx'):
                self.export_to_excel(filename)
            else:
                self.export_to_text(filename)

            messagebox.showinfo("Sukses", f"Hasil berhasil diekspor ke: {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengekspor hasil: {str(e)}")

    def export_to_csv(self, filename):
        """Export results to CSV"""
        import csv

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            headers = ['Model', 'Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE',
                       'Train_MAE', 'Test_MAE', 'Train_MAPE', 'Test_MAPE',
                       'CV_Mean', 'CV_Std', 'Training_Time']
            writer.writerow(headers)

            # Data
            for model_name, result in self.comparison_results.items():
                row = [
                    model_name.upper(),
                    f"{result['train_r2']:.4f}",
                    f"{result['test_r2']:.4f}",
                    f"{result['train_rmse']:.2f}",
                    f"{result['test_rmse']:.2f}",
                    f"{result['train_mae']:.2f}",
                    f"{result['test_mae']:.2f}",
                    f"{result['train_mape']:.2f}",
                    f"{result['test_mape']:.2f}",
                    f"{result['cv_mean']:.4f}" if result['cv_mean'] > 0 else "N/A",
                    f"{result['cv_std']:.4f}" if result['cv_std'] > 0 else "N/A",
                    f"{result['training_time']:.4f}"
                ]
                writer.writerow(row)

    def export_to_text(self, filename):
        """Export results to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            # Write summary text
            f.write(self.summary_text.get(1.0, tk.END))

    def reset_model_comparison(self):
        """Reset model comparison results"""
        self.comparison_results = {}

        # Clear table
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)

        # Clear summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Pilih model dan klik 'Jalankan Perbandingan' untuk memulai.")
        self.summary_text.config(state=tk.DISABLED)

        # Clear plot
        self.gambar_comparison.clear()
        self.kanvas_comparison.draw()

        self.reset_progress()

    def save_comparison_plot(self):
        """Save comparison plot"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
            )
            if filename:
                self.gambar_comparison.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Sukses", f"Plot berhasil disimpan: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan plot: {str(e)}")

    # Additional Excel export method (requires openpyxl)
    def export_to_excel(self, filename):
        """Export results to Excel (requires openpyxl)"""
        try:
            import pandas as pd

            # Prepare data
            data = []
            for model_name, result in self.comparison_results.items():
                data.append({
                    'Model': model_name.upper(),
                    'Train_R2': result['train_r2'],
                    'Test_R2': result['test_r2'],
                    'Train_RMSE': result['train_rmse'],
                    'Test_RMSE': result['test_rmse'],
                    'Train_MAE': result['train_mae'],
                    'Test_MAE': result['test_mae'],
                    'Train_MAPE': result['train_mape'],
                    'Test_MAPE': result['test_mape'],
                    'CV_Mean': result['cv_mean'] if result['cv_mean'] > 0 else None,
                    'CV_Std': result['cv_std'] if result['cv_std'] > 0 else None,
                    'Training_Time': result['training_time']
                })

            df = pd.DataFrame(data)
            df.to_excel(filename, index=False, sheet_name='Model_Comparison')

        except ImportError:
            # Fallback to CSV if pandas/openpyxl not available
            csv_filename = filename.replace('.xlsx', '.csv')
            self.export_to_csv(csv_filename)
            messagebox.showinfo("Info", f"Excel tidak tersedia, file disimpan sebagai CSV: {csv_filename}")
        except Exception as e:
            raise e

    def buat_tab_backtesting(self):
        """Membuat tab backtesting"""
        bingkai = ttk.Frame(self.tab_backtesting)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(bingkai, text="Tab Backtesting - Akan diimplementasikan",
                  font=("Arial", 14)).pack(pady=50)

    def buat_tab_teori(self):
        """Membuat tab teori dengan panduan yang diperluas dan estetik"""
        bingkai = ttk.Frame(self.tab_teori)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header dengan judul dan navigasi
        header_frame = ttk.Frame(bingkai)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = ttk.Label(header_frame, text="📚 Teori & Panduan Aplikasi",
                                font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)

        # Search box
        search_frame = ttk.Frame(header_frame)
        search_frame.pack(side=tk.RIGHT)

        ttk.Label(search_frame, text="🔍 Cari:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(0, 5))
        search_entry.bind('<KeyRelease>', self.search_theory_content)

        ttk.Button(search_frame, text="🔄", command=self.reset_theory_search, width=3).pack(side=tk.LEFT)

        # Main content menggunakan PanedWindow
        main_paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Table of Contents
        toc_frame = ttk.LabelFrame(main_paned, text="📑 Daftar Isi", width=250)
        main_paned.add(toc_frame, weight=1)

        # Table of Contents Treeview
        self.toc_tree = ttk.Treeview(toc_frame, show='tree')
        toc_scroll = ttk.Scrollbar(toc_frame, orient=tk.VERTICAL, command=self.toc_tree.yview)
        self.toc_tree.configure(yscrollcommand=toc_scroll.set)

        self.toc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        toc_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate table of contents
        self.setup_table_of_contents()

        # Bind selection event
        self.toc_tree.bind('<<TreeviewSelect>>', self.on_toc_selection)

        # Right panel - Content dengan Notebook
        content_frame = ttk.Frame(main_paned)
        main_paned.add(content_frame, weight=3)

        # Control panel untuk konten
        control_panel = ttk.Frame(content_frame)
        control_panel.pack(fill=tk.X, pady=(0, 5))

        # Font size controls
        font_frame = ttk.Frame(control_panel)
        font_frame.pack(side=tk.LEFT)

        ttk.Label(font_frame, text="📝 Font:").pack(side=tk.LEFT, padx=(0, 5))
        self.font_size = tk.IntVar(value=10)
        font_spinbox = ttk.Spinbox(font_frame, from_=8, to=16, textvariable=self.font_size,
                                   width=5, command=self.update_font_size)
        font_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # Theme toggle
        theme_frame = ttk.Frame(control_panel)
        theme_frame.pack(side=tk.LEFT)

        self.dark_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(theme_frame, text="🌙 Mode Gelap", variable=self.dark_mode,
                        command=self.toggle_content_theme).pack(side=tk.LEFT, padx=(0, 10))

        # Export button
        ttk.Button(control_panel, text="📤 Export PDF",
                   command=self.export_theory_pdf).pack(side=tk.RIGHT)

        # Content notebook
        self.theory_notebook = ttk.Notebook(content_frame)
        self.theory_notebook.pack(fill=tk.BOTH, expand=True)

        # Create theory tabs
        self.create_theory_tabs()

        # Status bar untuk teori
        status_frame = ttk.Frame(content_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.theory_status = ttk.Label(status_frame, text="Siap - Pilih topik dari daftar isi",
                                       font=("Arial", 9))
        self.theory_status.pack(side=tk.LEFT)

    def setup_table_of_contents(self):
        """Setup table of contents dengan struktur hierarkis"""
        toc_items = [
            ("📖 Pengantar", "intro", [
                ("🎯 Tujuan Aplikasi", "intro_purpose"),
                ("🔧 Cara Penggunaan", "intro_usage"),
                ("📊 Fitur Utama", "intro_features")
            ]),
            ("🔬 Teori SVD", "svd_theory", [
                ("📐 Dasar Matematika", "svd_math"),
                ("🧮 Algoritma SVD", "svd_algorithm"),
                ("💡 Implementasi", "svd_implementation"),
                ("✅ Keunggulan & Kelemahan", "svd_pros_cons")
            ]),
            ("🤖 Machine Learning Models", "ml_models", [
                ("📈 Linear Regression", "linear_regression"),
                ("🎯 Ridge Regression", "ridge_regression"),
                ("🔍 Lasso Regression", "lasso_regression"),
                ("🌳 Random Forest", "random_forest"),
                ("⚖️ Model Comparison", "model_comparison")
            ]),
            ("📊 Analisis Data", "data_analysis", [
                ("📈 Analisis Trend", "trend_analysis"),
                ("🔗 Analisis Korelasi", "correlation_analysis"),
                ("🎯 Analisis Sensitivitas", "sensitivity_analysis"),
                ("📉 Analisis Outlier", "outlier_analysis"),
                ("🎲 Monte Carlo Simulation", "monte_carlo")
            ]),
            ("📋 Panduan Penggunaan", "user_guide", [
                ("🗂️ Tab Data Historis", "guide_data"),
                ("🔮 Tab Prediksi", "guide_prediction"),
                ("📊 Tab Analisis Lanjutan", "guide_analysis"),
                ("🔬 Tab Perbandingan Model", "guide_comparison"),
                ("⚙️ Tips & Trik", "guide_tips")
            ]),
            ("❓ FAQ", "faq", [
                ("🤔 Pertanyaan Umum", "faq_general"),
                ("🐛 Troubleshooting", "faq_troubleshooting"),
                ("📞 Dukungan", "faq_support")
            ]),
            ("📚 Referensi", "references", [
                ("📖 Pustaka Ilmiah", "references_papers"),
                ("💻 Dokumentasi Teknis", "references_technical"),
                ("🔗 Sumber Online", "references_online")
            ])
        ]

        # Clear existing items
        for item in self.toc_tree.get_children():
            self.toc_tree.delete(item)

        # Add items to tree
        for title, key, children in toc_items:
            parent = self.toc_tree.insert("", "end", text=title, values=(key,))
            for child_title, child_key in children:
                self.toc_tree.insert(parent, "end", text=child_title, values=(child_key,))

        # Expand all items
        for item in self.toc_tree.get_children():
            self.toc_tree.item(item, open=True)

    def create_theory_tabs(self):
        """Membuat tabs untuk konten teori"""
        # Tab 1: Pengantar & Overview
        self.intro_tab = ttk.Frame(self.theory_notebook)
        self.theory_notebook.add(self.intro_tab, text="📖 Pengantar")
        self.create_intro_content()

        # Tab 2: Teori Matematika
        self.math_tab = ttk.Frame(self.theory_notebook)
        self.theory_notebook.add(self.math_tab, text="🔬 Teori SVD")
        self.create_math_content()

        # Tab 3: Machine Learning
        self.ml_tab = ttk.Frame(self.theory_notebook)
        self.theory_notebook.add(self.ml_tab, text="🤖 ML Models")
        self.create_ml_content()

        # Tab 4: Panduan Praktis
        self.guide_tab = ttk.Frame(self.theory_notebook)
        self.theory_notebook.add(self.guide_tab, text="📋 Panduan")
        self.create_guide_content()

        # Tab 5: FAQ & Troubleshooting
        self.faq_tab = ttk.Frame(self.theory_notebook)
        self.theory_notebook.add(self.faq_tab, text="❓ FAQ")
        self.create_faq_content()

    def create_intro_content(self):
        """Membuat konten pengantar"""
        # Create scrollable text widget
        text_frame = ttk.Frame(self.intro_tab)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.intro_text = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                                  bg="white", fg="black", padx=20, pady=20)

        intro_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.intro_text.yview)
        self.intro_text.configure(yscrollcommand=intro_scroll.set)

        self.intro_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        intro_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Content for intro
        intro_content = """
    🏆 APLIKASI PREDIKSI HARGA EMAS - VERSI DITINGKATKAN

    ═══════════════════════════════════════════════════════════════

    🎯 TUJUAN APLIKASI

    Aplikasi ini dirancang untuk membantu Anda melakukan prediksi harga emas menggunakan berbagai teknik analisis data modern, termasuk:

    • Singular Value Decomposition (SVD)
    • Machine Learning Algorithms (Linear, Ridge, Lasso, Random Forest)
    • Analisis Statistik Lanjutan
    • Monte Carlo Simulation
    • Cross-validation dan Backtesting

    ═══════════════════════════════════════════════════════════════

    🔧 CARA PENGGUNAAN CEPAT

    1. 📊 DATA HISTORIS
       → Import atau edit data historis harga emas
       → Verifikasi kualitas data dengan statistik dan visualisasi
       → Tambah/hapus data point sesuai kebutuhan

    2. 🔮 PREDIKSI
       → Masukkan parameter ekonomi (inflasi, suku bunga, indeks USD)
       → Pilih model prediksi (SVD, Linear, Ridge, Lasso, Random Forest)
       → Klik "Hitung Prediksi" untuk mendapatkan hasil

    3. 📈 ANALISIS LANJUTAN
       → Jalankan berbagai jenis analisis (korelasi, sensitivitas, trend)
       → Visualisasikan hasil dengan grafik interaktif
       → Export hasil analisis ke berbagai format

    4. 🔬 PERBANDINGAN MODEL
       → Bandingkan performa berbagai model ML
       → Lihat metrics lengkap (R², RMSE, MAE, MAPE)
       → Analisis residuals dan feature importance

    ═══════════════════════════════════════════════════════════════

    📊 FITUR UTAMA

    🔍 ANALISIS DATA KOMPREHENSIF
    • Statistik deskriptif lengkap
    • Deteksi outlier dengan berbagai metode
    • Analisis trend dan seasonality
    • Uji normalitas dan stationaritas

    🤖 MULTIPLE ML ALGORITHMS
    • SVD (Singular Value Decomposition)
    • Linear Regression dengan regularization
    • Ensemble methods (Random Forest)
    • Cross-validation untuk validasi model

    📈 VISUALISASI INTERAKTIF
    • Grafik dinamis dan zoomable
    • Multiple plot types (line, bar, scatter, area)
    • 3D visualization untuk surface plots
    • Export grafik dalam format HD

    🎯 ANALISIS PREDIKTIF
    • Confidence intervals untuk prediksi
    • Sensitivity analysis untuk parameter
    • Monte Carlo simulation untuk risk assessment
    • Backtesting untuk validasi historical

    ⚙️ ADVANCED FEATURES
    • Auto-save dan session management
    • Customizable themes (light/dark mode)
    • Export ke multiple formats (PDF, Excel, CSV)
    • Keyboard shortcuts untuk efficiency

    ═══════════════════════════════════════════════════════════════

    💡 TIPS PENGGUNAAN

    ✅ Selalu verifikasi kualitas data sebelum prediksi
    ✅ Gunakan multiple models untuk cross-validation
    ✅ Perhatikan confidence intervals dalam interpretasi
    ✅ Jalankan sensitivity analysis untuk robustness
    ✅ Export hasil untuk dokumentasi dan presentasi

    ═══════════════════════════════════════════════════════════════

    ⚠️ DISCLAIMER

    Aplikasi ini adalah tools analisis dan tidak memberikan nasihat investasi. 
    Selalu konsultasikan dengan ahli keuangan sebelum membuat keputusan investasi.
    Past performance is not indicative of future results.

    ═══════════════════════════════════════════════════════════════
    """

        # Configure text tags for styling
        self.intro_text.tag_configure("title", font=("Arial", 14, "bold"), foreground="darkblue")
        self.intro_text.tag_configure("section", font=("Arial", 12, "bold"), foreground="darkgreen")
        self.intro_text.tag_configure("subsection", font=("Arial", 11, "bold"), foreground="darkorange")
        self.intro_text.tag_configure("bullet", font=("Arial", 10), foreground="darkred")
        self.intro_text.tag_configure("tip", font=("Arial", 10), foreground="darkviolet")
        self.intro_text.tag_configure("warning", font=("Arial", 10, "italic"), foreground="red")

        # Insert content and apply tags
        self.intro_text.insert(tk.END, intro_content)
        self.apply_text_styling(self.intro_text)
        self.intro_text.config(state=tk.DISABLED)

    def create_math_content(self):
        """Membuat konten teori matematika"""
        text_frame = ttk.Frame(self.math_tab)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.math_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10),
                                 bg="white", fg="black", padx=20, pady=20)

        math_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.math_text.yview)
        self.math_text.configure(yscrollcommand=math_scroll.set)

        self.math_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        math_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        math_content = """
    🔬 SINGULAR VALUE DECOMPOSITION (SVD) - TEORI LENGKAP

    ═══════════════════════════════════════════════════════════════

    📐 DASAR MATEMATIKA

    SVD adalah faktorisasi matriks yang mengdekomposisi matriks M (m×n) menjadi:

        M = U × Σ × V^T

    Dimana:
    • U: Matriks orthogonal (m×m) - Left singular vectors
    • Σ: Matriks diagonal (m×n) - Singular values
    • V^T: Transpose matriks orthogonal (n×n) - Right singular vectors

    ───────────────────────────────────────────────────────────────

    🧮 PROPERTI MATEMATIKA

    1. ORTHOGONALITAS
       U^T × U = I    dan    V^T × V = I

    2. SINGULAR VALUES
       σ₁ ≥ σ₂ ≥ ... ≥ σₙ ≥ 0

    3. RANK APPROXIMATION
       M ≈ Σᵢ₌₁ʳ σᵢ × uᵢ × vᵢ^T

    ───────────────────────────────────────────────────────────────

    💻 IMPLEMENTASI DALAM PREDIKSI EMAS

    INPUT MATRIX:
    ┌                                    ┐
    │ inflasi₁    suku_bunga₁   usd₁     │
    │ inflasi₂    suku_bunga₂   usd₂     │  = M
    │    ...          ...       ...      │
    │ inflasiₙ    suku_bungaₙ   usdₙ     │
    └                                    ┘

    TARGET VECTOR:
    ┌            ┐
    │ harga_emas₁│
    │ harga_emas₂│  = y
    │    ...     │
    │ harga_emasₙ│
    └            ┘

    LANGKAH ALGORITMA:

    1. DEKOMPOSISI SVD
       [U, Σ, V^T] = SVD(M)

    2. HITUNG PSEUDO-INVERSE
       Σ⁺ = diag(1/σᵢ) untuk σᵢ > threshold

    3. HITUNG KOEFISIEN
       ω = V × Σ⁺ × U^T × y

    4. PREDIKSI
       y_pred = M_future × ω

    ───────────────────────────────────────────────────────────────

    🎯 KEUNGGULAN SVD

    ✅ STABILITAS NUMERIK
       • Tidak menggunakan matrix inversion langsung
       • Robust terhadap ill-conditioned matrices
       • Handling singular/near-singular matrices

    ✅ OPTIMAL SOLUTION
       • Memberikan least squares solution yang optimal
       • Minimizes ||Mω - y||² dalam norm Euclidean

    ✅ DIMENSIONALITY REDUCTION
       • Dapat memfilter noise dengan truncated SVD
       • Mengidentifikasi principal components

    ✅ MULTICOLLINEARITY HANDLING
       • Dapat menangani korelasi tinggi antar variabel
       • Tidak terpengaruh oleh redundant features

    ───────────────────────────────────────────────────────────────

    ⚠️ KELEMAHAN SVD

    ❌ COMPUTATIONAL COMPLEXITY
       • O(mn²) untuk m≥n atau O(m²n) untuk m<n
       • Intensive untuk large datasets

    ❌ INTERPRETABILITY
       • Koefisien ω tidak mudah diinterpretasi
       • Singular vectors tidak memiliki makna ekonomi langsung

    ❌ OVERFITTING RISK
       • Dapat overfit pada small datasets
       • Perlu regularization atau cross-validation

    ───────────────────────────────────────────────────────────────

    🔢 CONTOH NUMERIK

    Misalkan M = ┌ 2.75  3.50  100.7 ┐    y = ┌ 1773.3 ┐
                │ 3.12  3.75   95.6 │        │ 1807.2 │
                │ 3.36  4.00  104.1 │        │ 1806.9 │
                │ 1.80  5.81  103.5 │        │ 1962.2 │
                └ 2.09  6.10  104.5 ┘        └ 2416.4 ┘

    Setelah SVD:
    U ≈ ┌ -0.32  -0.59   0.65   0.33 ┐
        │ -0.31  -0.46  -0.71  -0.43 │
        │ -0.34  -0.48   0.25  -0.75 │
        │ -0.45   0.30   0.11   0.83 │
        └ -0.67   0.41  -0.10  -0.61 ┘

    Σ ≈ [108.2, 4.8, 2.1]

    V^T ≈ ┌ -0.05  -0.15  -0.99 ┐
          │  0.87  -0.49   0.04 │
          └ -0.49  -0.86   0.13 ┘

    ω ≈ ┌  23.45 ┐
        │ -187.32│
        └  15.67 ┘

    ───────────────────────────────────────────────────────────────

    📊 VALIDASI MODEL

    1. R-SQUARED (R²)
       R² = 1 - (SS_res / SS_tot)

    2. ROOT MEAN SQUARE ERROR (RMSE)
       RMSE = √(Σ(yᵢ - ŷᵢ)² / n)

    3. MEAN ABSOLUTE ERROR (MAE)
       MAE = Σ|yᵢ - ŷᵢ| / n

    4. CROSS-VALIDATION SCORE
       CV = (1/k)Σᵢ₌₁ᵏ Score(foldᵢ)

    ═══════════════════════════════════════════════════════════════
    """

        self.math_text.insert(tk.END, math_content)
        self.apply_text_styling(self.math_text)
        self.math_text.config(state=tk.DISABLED)

    def create_ml_content(self):
        """Membuat konten machine learning models"""
        text_frame = ttk.Frame(self.ml_tab)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.ml_text = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                               bg="white", fg="black", padx=20, pady=20)

        ml_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.ml_text.yview)
        self.ml_text.configure(yscrollcommand=ml_scroll.set)

        self.ml_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ml_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ml_content = """
    🤖 MACHINE LEARNING MODELS - PANDUAN LENGKAP

    ═══════════════════════════════════════════════════════════════

    📈 LINEAR REGRESSION

    KONSEP:
    Model paling sederhana yang mencari hubungan linear antara variabel independen (X) 
    dengan variabel dependen (y).

    FORMULA:
    y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ε

    Dimana:
    • β₀: Intercept
    • βᵢ: Koefisien untuk variabel xᵢ
    • ε: Error term

    KEUNGGULAN:
    ✅ Interpretable - mudah dipahami
    ✅ Fast computation - sangat cepat
    ✅ No hyperparameter tuning
    ✅ Good baseline model

    KELEMAHAN:
    ❌ Assumes linear relationship
    ❌ Sensitive to outliers
    ❌ No regularization - prone to overfitting
    ❌ Multicollinearity issues

    KAPAN DIGUNAKAN:
    • Dataset kecil dengan hubungan linear jelas
    • Butuh interpretabilitas tinggi
    • Sebagai baseline comparison

    ───────────────────────────────────────────────────────────────

    🎯 RIDGE REGRESSION (L2 Regularization)

    KONSEP:
    Linear regression dengan penalty L2 untuk mencegah overfitting dan menangani 
    multicollinearity.

    FORMULA:
    Loss = MSE + α × Σβᵢ²

    Dimana α adalah regularization parameter.

    KEUNGGULAN:
    ✅ Handles multicollinearity well
    ✅ Prevents overfitting
    ✅ Stable coefficients
    ✅ Works with p > n scenarios

    KELEMAHAN:
    ❌ Doesn't perform feature selection
    ❌ All features retained with small coefficients
    ❌ Need to tune α parameter
    ❌ Less interpretable than linear regression

    HYPERPARAMETER:
    • α (alpha): Regularization strength
      - α = 0: sama dengan linear regression
      - α → ∞: coefficients → 0
      - Typical range: 0.01 to 100

    KAPAN DIGUNAKAN:
    • Dataset dengan multicollinearity
    • Lebih banyak features daripada samples
    • Butuh model yang stable

    ───────────────────────────────────────────────────────────────

    🔍 LASSO REGRESSION (L1 Regularization)

    KONSEP:
    Linear regression dengan penalty L1 yang dapat melakukan automatic feature selection.

    FORMULA:
    Loss = MSE + α × Σ|βᵢ|

    KEUNGGULAN:
    ✅ Automatic feature selection
    ✅ Sparse solutions (some βᵢ = 0)
    ✅ Handles multicollinearity
    ✅ Interpretable results

    KELEMAHAN:
    ❌ Can arbitrarily select one feature from correlated group
    ❌ Unstable feature selection
    ❌ Less stable than Ridge
    ❌ May underperform with small α

    HYPERPARAMETER:
    • α (alpha): Regularization strength
      - Higher α → more features set to zero
      - Lower α → closer to linear regression

    KAPAN DIGUNAKAN:
    • Banyak features, hanya beberapa yang relevan
    • Butuh feature selection otomatis
    • Interpretabilitas penting

    ───────────────────────────────────────────────────────────────

    🌳 RANDOM FOREST

    KONSEP:
    Ensemble method yang menggabungkan multiple decision trees dengan voting/averaging.

    ALGORITMA:
    1. Bootstrap sampling dari training data
    2. Random feature selection di setiap split
    3. Build decision tree untuk setiap bootstrap sample
    4. Aggregate predictions (voting untuk classification, averaging untuk regression)

    KEUNGGULAN:
    ✅ Handles non-linear relationships
    ✅ Robust to outliers
    ✅ Provides feature importance
    ✅ No assumptions about data distribution
    ✅ Handles missing values
    ✅ Less prone to overfitting

    KELEMAHAN:
    ❌ Black box model - less interpretable
    ❌ Can overfit with very noisy data
    ❌ Biased towards categorical variables with more levels
    ❌ Larger memory footprint

    HYPERPARAMETERS:
    • n_estimators: Number of trees (default: 100)
    • max_depth: Maximum depth of trees
    • min_samples_split: Minimum samples to split node
    • min_samples_leaf: Minimum samples in leaf
    • max_features: Number of features for best split

    KAPAN DIGUNAKAN:
    • Complex non-linear relationships
    • Mixed data types (numerical + categorical)
    • Robust baseline model diperlukan
    • Feature importance analysis

    ───────────────────────────────────────────────────────────────

    ⚖️ MODEL COMPARISON FRAMEWORK

    METRICS EVALUASI:

    1. R² SCORE (Coefficient of Determination)
       • Range: -∞ to 1
       • 1 = perfect fit, 0 = no predictive power
       • < 0 = worse than predicting mean

    2. RMSE (Root Mean Square Error)
       • Same unit as target variable
       • Penalizes large errors more heavily
       • Lower is better

    3. MAE (Mean Absolute Error)
       • Same unit as target variable
       • Less sensitive to outliers than RMSE
       • Lower is better

    4. MAPE (Mean Absolute Percentage Error)
       • Percentage error - easy to interpret
       • Issues when actual values near zero
       • Lower is better

    VALIDATION STRATEGY:

    1. TRAIN/TEST SPLIT
       • Simple split (e.g., 80/20)
       • Stratified for balanced datasets

    2. CROSS-VALIDATION
       • K-fold CV (typically k=5 or k=10)
       • Leave-one-out CV for small datasets
       • Time Series CV for temporal data

    3. NESTED CV
       • Outer loop: model selection
       • Inner loop: hyperparameter tuning

    SELECTION CRITERIA:

    🏆 BEST MODEL SELECTION:
    1. Highest CV R² score
    2. Lowest CV RMSE
    3. Smallest gap between train/test performance
    4. Consistent performance across folds
    5. Computational efficiency
    6. Interpretability requirements

    ═══════════════════════════════════════════════════════════════

    💡 PRACTICAL RECOMMENDATIONS

    UNTUK PREDIKSI HARGA EMAS:

    1. START WITH SIMPLE
       • Begin with Linear Regression as baseline
       • Check for linear relationships in data

    2. HANDLE MULTICOLLINEARITY
       • Use Ridge if variables are correlated
       • Lasso if feature selection needed

    3. CAPTURE NON-LINEARITY
       • Random Forest for complex patterns
       • SVD for dimensionality reduction

    4. ENSEMBLE APPROACH
       • Combine predictions from multiple models
       • Weighted average based on CV performance

    5. VALIDATE THOROUGHLY
       • Use time series validation for temporal data
       • Check residuals for patterns
       • Test on out-of-sample data

    ═══════════════════════════════════════════════════════════════
    """

        self.ml_text.insert(tk.END, ml_content)
        self.apply_text_styling(self.ml_text)
        self.ml_text.config(state=tk.DISABLED)

    def create_guide_content(self):
        """Membuat konten panduan penggunaan"""
        text_frame = ttk.Frame(self.guide_tab)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.guide_text = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                                  bg="white", fg="black", padx=20, pady=20)

        guide_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.guide_text.yview)
        self.guide_text.configure(yscrollcommand=guide_scroll.set)

        self.guide_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        guide_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        guide_content = """
    📋 PANDUAN PENGGUNAAN LENGKAP

    ═══════════════════════════════════════════════════════════════

    🗂️ TAB DATA HISTORIS

    FUNGSI UTAMA:
    • Input dan edit data historis harga emas
    • Visualisasi data dengan berbagai jenis chart
    • Analisis statistik deskriptif
    • Import/export data dari/ke file

    CARA PENGGUNAAN:

    1. INPUT DATA
       ➤ Klik "➕ Tambah" untuk menambah data baru
       ➤ Klik "✏️ Edit" untuk mengubah data yang dipilih
       ➤ Klik "🗑️ Hapus" untuk menghapus data

    2. IMPORT DATA
       ➤ Klik "📥 Impor" → pilih file CSV/Excel
       ➤ Pastikan format: Tahun, Inflasi, Suku_Bunga, Indeks_USD, Harga_Emas
       ➤ Verifikasi data setelah import

    3. VISUALISASI
       ➤ Pilih jenis grafik: Line, Bar, Scatter, Area
       ➤ Pilih variabel: Semua atau individual
       ➤ Gunakan toolbar untuk zoom, pan, save

    4. ANALISIS STATISTIK
       ➤ Panel kanan menampilkan mean, std, min, max
       ➤ Gunakan untuk deteksi outlier dan kualitas data

    TIPS:
    💡 Selalu verifikasi data sebelum analisis
    💡 Gunakan scatter plot untuk melihat korelasi
    💡 Export data backup secara berkala

    ───────────────────────────────────────────────────────────────

    🔮 TAB PREDIKSI

    FUNGSI UTAMA:
    • Input parameter ekonomi untuk prediksi
    • Pilih model prediksi (SVD, Linear, Ridge, Lasso, Random Forest)
    • Lihat hasil prediksi dengan confidence interval
    • Analisis komponen SVD dan feature importance

    LANGKAH-LANGKAH:

    1. INPUT PARAMETER
       ➤ Tahun: Tahun target prediksi
       ➤ Inflasi (%): Perkiraan tingkat inflasi
       ➤ Suku Bunga (%): Suku bunga acuan yang diharapkan
       ➤ Indeks USD: Proyeksi kekuatan dollar AS

    2. PILIH MODEL
       ➤ SVD: Robust, good for small datasets
       ➤ Linear: Simple, interpretable baseline
       ➤ Ridge: Good for correlated features
       ➤ Lasso: Automatic feature selection
       ➤ Random Forest: Handles non-linearity

    3. OPSI LANJUTAN
       ☑️ Confidence Interval: Untuk uncertainty quantification
       ☑️ Feature Importance: Untuk melihat kontribusi variabel
       ☑️ Residual Analysis: Untuk validasi model

    4. JALANKAN PREDIKSI
       ➤ Klik "🔮 Hitung Prediksi"
       ➤ Tunggu proses selesai
       ➤ Lihat hasil di panel kanan

    INTERPRETASI HASIL:
    • Harga Prediksi: Estimasi harga emas dalam USD
    • Confidence Interval: Range kemungkinan nilai
    • Feature Importance: Kontribusi relatif setiap variabel
    • R² Score: Goodness of fit model (0-1, higher is better)

    ───────────────────────────────────────────────────────────────

    📊 TAB ANALISIS LANJUTAN

    FUNGSI UTAMA:
    • Berbagai jenis analisis statistik dan ekonometrik
    • Visualisasi advanced dengan plot interaktif
    • Export hasil analisis

    JENIS ANALISIS:

    1. ANALISIS KORELASI
       ➤ Melihat hubungan antar variabel ekonomi
       ➤ Heatmap correlation matrix dengan color coding
       ➤ Significance testing dengan p-values
       ➤ Interpretasi: nilai mendekati +1 (korelasi positif kuat), 
         mendekati -1 (korelasi negatif kuat), mendekati 0 (tidak berkorelasi)

    2. ANALISIS SENSITIVITAS
       ➤ Dampak perubahan parameter terhadap prediksi harga emas
       ➤ Variasi parameter ±5%, ±10%, ±20% dari nilai base
       ➤ Ranking sensitivitas variabel (mana yang paling berpengaruh)
       ➤ Scenario analysis untuk risk assessment
       ➤ Interpretasi: variabel dengan sensitivitas tinggi memerlukan perhatian khusus

    3. ANALISIS TREND
       ➤ Linear trend untuk setiap variabel ekonomi
       ➤ Growth rate calculation (persentase perubahan year-over-year)
       ➤ Trend strength dan statistical significance (R², p-value)
       ➤ Moving average untuk smoothing data
       ➤ Interpretasi: trend positif menunjukkan kenaikan, negatif menunjukkan penurunan

    4. ANALISIS OUTLIER
       ➤ Deteksi data yang tidak biasa atau ekstrem
       ➤ Metode Z-score: |z| > 2 dianggap outlier
       ➤ Metode IQR: data di luar Q1-1.5*IQR atau Q3+1.5*IQR
       ➤ Modified Z-score (MAD): lebih robust terhadap outlier
       ➤ Visualisasi outliers dengan scatter plot dan highlighting
       ➤ Impact assessment: pengaruh outlier terhadap model

    5. TIME SERIES ANALYSIS
       ➤ Decomposition: trend, seasonal, residual components
       ➤ Stationarity testing untuk validasi asumsi time series
       ➤ Autocorrelation analysis untuk pola temporal
       ➤ Interpretasi komponen:
         - Trend: arah pergerakan jangka panjang
         - Seasonal: pola berulang (jika ada)
         - Residual: noise yang tidak dapat dijelaskan

    6. STATISTICAL TESTING
       ➤ Normality tests (Shapiro-Wilk): apakah data berdistribusi normal
       ➤ Confidence intervals untuk mean setiap variabel
       ➤ Correlation significance testing
       ➤ Descriptive statistics: mean, median, std, skewness, kurtosis
       ➤ Interpretasi p-values: < 0.05 signifikan, < 0.01 sangat signifikan

    7. MONTE CARLO SIMULATION
       ➤ Risk assessment dengan random sampling dari distribusi parameter
       ➤ Generate ribuan skenario prediksi berdasarkan uncertainty
       ➤ Distribution of predictions untuk melihat range kemungkinan
       ➤ Risk metrics:
         - VaR (Value at Risk): worst-case scenario pada confidence level tertentu
         - CVaR (Conditional VaR): rata-rata kerugian di atas VaR
         - Probability of loss: kemungkinan harga turun dari ekspektasi
       ➤ Visualization: histogram, box plot, cumulative distribution

    8. FEATURE ANALYSIS
       ➤ Feature engineering: membuat variabel baru dari kombinasi existing features
       ➤ Feature importance dengan multiple methods:
         - Correlation-based: korelasi absolut dengan target
         - Mutual information: dependency non-linear
         - F-statistics: statistical significance
         - Random Forest importance: contribution dalam ensemble model
       ➤ Feature selection recommendations
       ➤ Interaction effects analysis

    CARA PENGGUNAAN:
    1. Pilih jenis analisis dari radio button
    2. Set parameter sesuai kebutuhan
    3. Klik "🔍 Jalankan Analisis"
    4. Lihat hasil di text panel dan visualisasi
    5. Export hasil jika diperlukan

    ───────────────────────────────────────────────────────────────

    🔬 TAB PERBANDINGAN MODEL

    FUNGSI UTAMA:
    • Benchmark multiple models secara bersamaan
    • Cross-validation dan statistical testing
    • Comprehensive performance metrics
    • Model selection recommendations

    PENGATURAN:

    1. MODEL SELECTION
       ☑️ Pilih model yang ingin dibandingkan
       ☑️ Minimal 2 model untuk comparison

    2. VALIDATION SETTINGS
       ➤ CV Folds: Jumlah fold untuk cross-validation (3-10)
       ➤ Test Split: Proporsi data untuk testing (0.1-0.5)
       ➤ Random State: Seed untuk reproducibility

    3. COMPARISON OPTIONS
       ☑️ Feature Importance Analysis
       ☑️ Residuals Analysis
       ☑️ Statistical Tests
       ☑️ Training Time Analysis

    HASIL YANG DITAMPILKAN:

    1. TABEL COMPARISON
       • Model ranking berdasarkan metrics
       • R², RMSE, MAE, MAPE scores
       • Cross-validation statistics

    2. VISUALISASI
       • Metrics comparison (bar charts)
       • Actual vs Predicted scatter plots
       • Residuals analysis
       • Feature importance comparison
       • CV scores distribution

    3. STATISTICAL ANALYSIS
       • Model significance testing
       • Bias-variance analysis
       • Performance consistency

    4. RECOMMENDATIONS
       • Best model selection dengan reasoning
       • Overfitting detection
       • Performance interpretation

    ───────────────────────────────────────────────────────────────

    ⚙️ TIPS & TRIK PENGGUNAAN

    WORKFLOW OPTIMAL:

    1. 📊 PERSIAPAN DATA
       ✓ Import data historis yang lengkap dan akurat
       ✓ Verifikasi tidak ada missing values
       ✓ Check outliers dan data quality
       ✓ Backup data original

    2. 🔍 EXPLORATORY ANALYSIS
       ✓ Lihat distribusi dan statistik setiap variabel
       ✓ Analisis korelasi antar variabel
       ✓ Deteksi trend dan seasonality
       ✓ Identifikasi outliers

    3. 🤖 MODEL DEVELOPMENT
       ✓ Mulai dengan baseline model (Linear Regression)
       ✓ Test multiple models dengan cross-validation
       ✓ Tune hyperparameters jika diperlukan
       ✓ Validate dengan out-of-sample data

    4. 📈 MODEL EVALUATION
       ✓ Compare multiple metrics (R², RMSE, MAE)
       ✓ Check residuals untuk patterns
       ✓ Analyze feature importance
       ✓ Test model stability dengan bootstrap

    5. 🎯 PREDICTION & INTERPRETATION
       ✓ Use best model untuk final prediction
       ✓ Calculate confidence intervals
       ✓ Perform sensitivity analysis
       ✓ Document assumptions dan limitations

    KEYBOARD SHORTCUTS:
    • Ctrl+O: Import data
    • Ctrl+S: Export results
    • Ctrl+R: Run prediction
    • F5: Refresh data
    • Ctrl+T: Toggle theme

    TROUBLESHOOTING UMUM:

    ❌ PREDIKSI TIDAK AKURAT
       ➤ Check data quality dan completeness
       ➤ Coba model yang berbeda
       ➤ Increase dataset size jika memungkinkan
       ➤ Consider additional features

    ❌ MODEL OVERFITTING
       ➤ Use regularization (Ridge/Lasso)
       ➤ Reduce model complexity
       ➤ Increase training data
       ➤ Use cross-validation

    ❌ SLOW PERFORMANCE
       ➤ Reduce dataset size
       ➤ Use simpler models
       ➤ Enable caching
       ➤ Close unnecessary applications

    ═══════════════════════════════════════════════════════════════
    """

        self.guide_text.insert(tk.END, guide_content)
        self.apply_text_styling(self.guide_text)
        self.guide_text.config(state=tk.DISABLED)

    def create_faq_content(self):
        """Membuat konten FAQ"""
        text_frame = ttk.Frame(self.faq_tab)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.faq_text = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10),
                                bg="white", fg="black", padx=20, pady=20)

        faq_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.faq_text.yview)
        self.faq_text.configure(yscrollcommand=faq_scroll.set)

        self.faq_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        faq_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        faq_content = """
    ❓ FREQUENTLY ASKED QUESTIONS

    ═══════════════════════════════════════════════════════════════

    🤔 PERTANYAAN UMUM

    Q: Apakah aplikasi ini memberikan nasihat investasi?
    A: TIDAK. Aplikasi ini adalah tool analisis untuk educational purposes. 
       Selalu konsultasi dengan financial advisor sebelum membuat keputusan investasi.

    Q: Seberapa akurat prediksi harga emas dari aplikasi ini?
    A: Akurasi tergantung pada kualitas data historis dan kondisi pasar. 
       Prediksi financial selalu memiliki uncertainty. Gunakan confidence intervals 
       untuk memahami range kemungkinan nilai.

    Q: Model mana yang paling baik untuk prediksi harga emas?
    A: Tidak ada model yang universally "terbaik". Gunakan tab Model Comparison 
       untuk membandingkan performa berbagai model pada data Anda. 
       Umumnya Random Forest good untuk non-linear patterns, SVD robust untuk small datasets.

    Q: Berapa banyak data historis yang diperlukan?
    A: Minimum 5-10 data points, idealnya 20+ untuk hasil yang lebih reliable. 
       Semakin banyak data berkualitas, semakin baik prediksi.

    Q: Bisakah menambahkan variabel ekonomi lain?
    A: Saat ini aplikasi support 3 variabel input: inflasi, suku bunga, indeks USD. 
       Untuk additional features, perlu modifikasi code.

    Q: Apakah bisa memprediksi untuk beberapa tahun ke depan?
    A: Ya, tapi akurasi menurun untuk horizon yang lebih jauh. 
       Recommended untuk prediksi 1-2 tahun ke depan.

    ───────────────────────────────────────────────────────────────

    🐛 TROUBLESHOOTING

    Q: Error "Input tidak valid" saat memasukkan parameter
    A: • Pastikan menggunakan format angka yang benar (gunakan titik untuk decimal)
       • Check range values yang reasonable (inflasi 0-20%, suku bunga 0-25%, dll)
       • Jangan gunakan karakter khusus atau huruf

    Q: Aplikasi menjadi lambat atau hang
    A: • Tutup tab browser/aplikasi lain yang tidak diperlukan
       • Reduce number of Monte Carlo simulations
       • Clear cache dari menu Edit → Reset Cache
       • Restart aplikasi jika perlu

    Q: Grafik tidak muncul atau error visualisasi
    A: • Check apakah data historis sudah diinput dengan benar
       • Refresh visualisasi dengan tombol 🔄
       • Coba ganti jenis plot (line, bar, scatter)
       • Reset zoom jika grafik terpotong

    Q: Import data CSV/Excel tidak berhasil
    A: • Pastikan format file: Tahun, Inflasi, Suku_Bunga, Indeks_USD, Harga_Emas
       • Header harus exact match (case sensitive)
       • Tidak ada missing values atau cells kosong
       • Gunakan format CSV dengan comma separator

    Q: Prediksi menghasilkan nilai yang tidak masuk akal
    A: • Verifikasi input parameters dalam range yang reasonable
       • Check data historis untuk outliers atau errors
       • Coba model yang berbeda
       • Lihat confidence intervals untuk uncertainty estimate

    Q: Cross-validation error atau model comparison gagal
    A: • Pastikan memiliki cukup data (minimal 5 points)
       • Check tidak ada missing values
       • Reduce CV folds jika dataset kecil
       • Disable advanced options jika ada error

    ───────────────────────────────────────────────────────────────

    📞 DUKUNGAN TEKNIS

    INFORMASI SISTEM:
    • Aplikasi: Prediksi Harga Emas v2.0
    • Platform: Windows/Mac/Linux
    • Dependencies: NumPy, Pandas, Scikit-learn, Matplotlib, Tkinter

    REPORTING BUGS:
    Jika menemukan bug atau error, catat informasi berikut:
    1. Langkah yang dilakukan sebelum error
    2. Pesan error yang muncul
    3. Data yang digunakan (jika bisa dishare)
    4. Screenshot error (jika relevan)

    FEATURE REQUESTS:
    Saran untuk improvement aplikasi:
    • Additional ML models (XGBoost, Neural Networks)
    • More economic indicators
    • Real-time data integration
    • Advanced visualization options
    • Export formats tambahan

    LEARNING RESOURCES:
    📚 Buku Referensi:
    • "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
    • "Pattern Recognition and Machine Learning" - Christopher Bishop
    • "Financial Econometrics" - Petr Zemčík

    🌐 Online Courses:
    • Coursera: Machine Learning by Andrew Ng
    • edX: Introduction to Computational Thinking and Data Science
    • Khan Academy: Statistics and Probability

    🔗 Technical Documentation:
    • NumPy: https://numpy.org/doc/
    • Pandas: https://pandas.pydata.org/docs/
    • Scikit-learn: https://scikit-learn.org/stable/
    • Matplotlib: https://matplotlib.org/stable/contents.html

    ───────────────────────────────────────────────────────────────

    ⚠️ DISCLAIMER & RISK WARNINGS

    INVESTMENT DISCLAIMER:
    • Past performance tidak mengindikasikan future results
    • Gold prices extremely volatile dan dipengaruhi banyak faktor
    • Economic indicators hanya sebagian dari factors yang mempengaruhi harga
    • Selalu diversifikasi portfolio dan jangan invest lebih dari yang Anda mampu

    TECHNICAL LIMITATIONS:
    • Model berdasarkan historical data yang terbatas
    • Tidak memperhitungkan black swan events
    • Assumptions tentang linear/predictable relationships
    • Market sentiment dan geopolitical factors tidak dimodelkan

    RECOMMENDED USAGE:
    • Gunakan sebagai educational tool
    • Combine dengan fundamental analysis
    • Consider multiple scenarios dan stress testing
    • Regular model retraining dengan data terbaru
    • Professional consultation untuk investment decisions

    ═══════════════════════════════════════════════════════════════

    📧 KONTAK & FEEDBACK

    Jika ada pertanyaan, saran, atau feedback, silakan hubungi pengembang 
    melalui channel yang tersedia dalam dokumentasi aplikasi.

    Terima kasih telah menggunakan Aplikasi Prediksi Harga Emas!

    ═══════════════════════════════════════════════════════════════
    """

        self.faq_text.insert(tk.END, faq_content)
        self.apply_text_styling(self.faq_text)
        self.faq_text.config(state=tk.DISABLED)

    def apply_text_styling(self, text_widget):
        """Apply consistent styling to text widgets"""
        # Configure tags
        text_widget.tag_configure("title", font=("Arial", 14, "bold"), foreground="darkblue")
        text_widget.tag_configure("section", font=("Arial", 12, "bold"), foreground="darkgreen")
        text_widget.tag_configure("subsection", font=("Arial", 11, "bold"), foreground="darkorange")
        text_widget.tag_configure("code", font=("Consolas", 9), background="lightgray")
        text_widget.tag_configure("emphasis", font=("Arial", 10, "bold"), foreground="darkred")
        text_widget.tag_configure("warning", font=("Arial", 10, "bold"), foreground="red")

    def on_toc_selection(self, event):
        """Handle table of contents selection"""
        selection = self.toc_tree.selection()
        if selection:
            item = self.toc_tree.item(selection[0])
            if item['values']:
                topic_key = item['values'][0]
                self.navigate_to_topic(topic_key)
                self.theory_status.config(text=f"Menampilkan: {item['text']}")

    def navigate_to_topic(self, topic_key):
        """Navigate to specific topic in theory content"""
        # Map topic keys to tab indices and search terms
        topic_mapping = {
            # Pengantar Tab (0)
            'intro': (0, None),
            'intro_purpose': (0, "TUJUAN APLIKASI"),
            'intro_usage': (0, "CARA PENGGUNAAN"),
            'intro_features': (0, "FITUR UTAMA"),

            # Teori SVD Tab (1)
            'svd_theory': (1, None),
            'svd_math': (1, "DASAR MATEMATIKA"),
            'svd_algorithm': (1, "ALGORITMA SVD"),
            'svd_implementation': (1, "IMPLEMENTASI DALAM PREDIKSI"),
            'svd_pros_cons': (1, "KEUNGGULAN SVD"),

            # ML Models Tab (2)
            'ml_models': (2, None),
            'linear_regression': (2, "LINEAR REGRESSION"),
            'ridge_regression': (2, "RIDGE REGRESSION"),
            'lasso_regression': (2, "LASSO REGRESSION"),
            'random_forest': (2, "RANDOM FOREST"),
            'model_comparison': (2, "MODEL COMPARISON FRAMEWORK"),

            # Data Analysis Topics - Map to Guide Tab (3) where analysis is explained
            'data_analysis': (3, "TAB ANALISIS LANJUTAN"),
            'trend_analysis': (3, "ANALISIS TREND"),
            'correlation_analysis': (3, "ANALISIS KORELASI"),
            'sensitivity_analysis': (3, "ANALISIS SENSITIVITAS"),
            'outlier_analysis': (3, "ANALISIS OUTLIER"),
            'monte_carlo': (3, "MONTE CARLO SIMULATION"),

            # User Guide Tab (3)
            'user_guide': (3, None),
            'guide_data': (3, "TAB DATA HISTORIS"),
            'guide_prediction': (3, "TAB PREDIKSI"),
            'guide_analysis': (3, "TAB ANALISIS LANJUTAN"),
            'guide_comparison': (3, "TAB PERBANDINGAN MODEL"),
            'guide_tips': (3, "TIPS & TRIK PENGGUNAAN"),

            # FAQ Tab (4)
            'faq': (4, None),
            'faq_general': (4, "PERTANYAAN UMUM"),
            'faq_troubleshooting': (4, "TROUBLESHOOTING"),
            'faq_support': (4, "DUKUNGAN TEKNIS"),

            # References - Map to FAQ tab for now
            'references': (4, "LEARNING RESOURCES"),
            'references_papers': (4, "Buku Referensi"),
            'references_technical': (4, "Technical Documentation"),
            'references_online': (4, "Online Courses")
        }

        if topic_key in topic_mapping:
            tab_index, search_term = topic_mapping[topic_key]

            # Switch to appropriate tab
            self.theory_notebook.select(tab_index)

            # Small delay to ensure tab is loaded
            self.root.after(100, lambda: self.delayed_scroll(tab_index, search_term))
        else:
            # Handle unmapped topics
            self.theory_status.config(text=f"Topik '{topic_key}' belum tersedia")

    def delayed_scroll(self, tab_index, search_term):
        """Delayed scroll to section after tab switch"""
        if search_term:
            self.scroll_to_section(tab_index, search_term)
        else:
            # Just move to top of the tab
            text_widgets = [self.intro_text, self.math_text, self.ml_text,
                            self.guide_text, self.faq_text]
            if tab_index < len(text_widgets):
                text_widgets[tab_index].see("1.0")

    def scroll_to_section(self, tab_index, search_term):
        """Scroll to specific section in text widget"""
        text_widgets = [self.intro_text, self.math_text, self.ml_text,
                        self.guide_text, self.faq_text]

        if tab_index >= len(text_widgets):
            return

        text_widget = text_widgets[tab_index]

        # Try different variations of the search term
        search_variations = [
            search_term,
            search_term.upper(),
            search_term.lower(),
            search_term.title(),
            search_term.replace(" ", "_"),
            search_term.replace("_", " ")
        ]

        found_pos = None
        for variation in search_variations:
            found_pos = text_widget.search(variation, "1.0", tk.END)
            if found_pos:
                break

        if found_pos:
            # Scroll to the found position
            text_widget.see(found_pos)

            # Highlight the found text temporarily
            end_pos = f"{found_pos}+{len(search_term)}c"
            text_widget.tag_add("highlight", found_pos, end_pos)
            text_widget.tag_configure("highlight", background="yellow", foreground="black")

            # Remove highlight after 3 seconds
            self.root.after(3000, lambda: text_widget.tag_remove("highlight", "1.0", tk.END))

            # Update status
            self.theory_status.config(text=f"Menampilkan: {search_term}")
        else:
            # If not found, just go to top and show a message
            text_widget.see("1.0")
            self.theory_status.config(text=f"Bagian '{search_term}' tidak ditemukan, menampilkan awal halaman")

    def search_theory_content(self, event=None):
        """Search content across all theory tabs"""
        search_term = self.search_var.get().lower()
        if not search_term:
            return

        # Search across all text widgets
        text_widgets = [
            (self.intro_text, "Pengantar"),
            (self.math_text, "Teori SVD"),
            (self.ml_text, "ML Models"),
            (self.guide_text, "Panduan"),
            (self.faq_text, "FAQ")
        ]

        results = []
        for text_widget, tab_name in text_widgets:
            try:
                if hasattr(self, text_widget.__class__.__name__.lower() + '_text') or hasattr(self,
                                                                                              text_widget.__class__.__name__):
                    content = text_widget.get("1.0", tk.END).lower()
                    if search_term in content:
                        results.append(tab_name)
            except Exception as e:
                print(f"Error searching in {tab_name}: {e}")
                continue

        if results:
            self.theory_status.config(text=f"Ditemukan di: {', '.join(results)}")
        else:
            self.theory_status.config(text=f"Tidak ditemukan: '{search_term}'")

    def debug_navigation(self):
        """Debug method to check navigation setup"""
        print("=== DEBUG NAVIGATION ===")

        # Check text widgets
        text_widgets = ['intro_text', 'math_text', 'ml_text', 'guide_text', 'faq_text']
        for widget_name in text_widgets:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                content_length = len(widget.get("1.0", tk.END))
                print(f"✓ {widget_name}: exists, content length: {content_length}")
            else:
                print(f"✗ {widget_name}: MISSING")

        # Check table of contents
        toc_items = self.toc_tree.get_children()
        print(f"TOC items count: {len(toc_items)}")

        for item in toc_items[:3]:  # Check first 3 items
            item_data = self.toc_tree.item(item)
            print(f"TOC item: {item_data['text']} -> {item_data['values']}")

            # Check children
            children = self.toc_tree.get_children(item)
            for child in children[:2]:  # Check first 2 children
                child_data = self.toc_tree.item(child)
                print(f"  Child: {child_data['text']} -> {child_data['values']}")

        print("=== END DEBUG ===")

    # Add a button in the control panel to call debug (for testing)
    def add_debug_button_to_control_panel(self, control_panel):
        """Add debug button to control panel (for development)"""
        debug_frame = ttk.Frame(control_panel)
        debug_frame.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(debug_frame, text="🐛 Debug",
                   command=self.debug_navigation, width=8).pack(side=tk.LEFT)

    def reset_theory_search(self):
        """Reset search and clear highlights"""
        self.search_var.set("")
        self.theory_status.config(text="Siap - Pilih topik dari daftar isi")

        # Clear all highlights
        text_widgets = [self.intro_text, self.math_text, self.ml_text,
                        self.guide_text, self.faq_text]
        for widget in text_widgets:
            widget.tag_remove("highlight", "1.0", tk.END)

    def update_font_size(self):
        """Update font size for all text widgets"""
        new_size = self.font_size.get()
        text_widgets = [self.intro_text, self.math_text, self.ml_text,
                        self.guide_text, self.faq_text]

        for widget in text_widgets:
            current_font = widget.cget("font")
            if isinstance(current_font, str):
                # Parse font string
                parts = current_font.split()
                font_family = parts[0] if parts else "Arial"
            else:
                font_family = current_font[0] if current_font else "Arial"

            widget.config(font=(font_family, new_size))

    def toggle_content_theme(self):
        """Toggle between light and dark theme for content"""
        if self.dark_mode.get():
            # Dark theme
            bg_color = "#2d2d2d"
            fg_color = "#ffffff"
            select_color = "#404040"
        else:
            # Light theme
            bg_color = "white"
            fg_color = "black"
            select_color = "#e0e0e0"

        text_widgets = [self.intro_text, self.math_text, self.ml_text,
                        self.guide_text, self.faq_text]

        for widget in text_widgets:
            widget.config(bg=bg_color, fg=fg_color, selectbackground=select_color)

    def export_theory_pdf(self):
        """Export theory content to PDF"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )

            if filename:
                # This would require implementing PDF generation
                # For now, show a placeholder message
                messagebox.showinfo("Export PDF",
                                    "Fitur export PDF akan diimplementasikan. "
                                    "Saat ini bisa copy-paste konten ke word processor.")
        except ImportError:
            messagebox.showwarning("Export PDF",
                                   "Library reportlab tidak tersedia. "
                                   "Install dengan: pip install reportlab")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal export PDF: {str(e)}")
            
    # === Utility Methods ===

    def buat_tooltip(self, widget, text):
        """Membuat tooltip untuk widget"""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = ttk.Label(tooltip, text=text, background="lightyellow",
                              relief="solid", borderwidth=1)
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def update_progress(self, value, status):
        """Update progress bar dan status"""
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update_idletasks()

    def reset_progress(self):
        """Reset progress bar"""
        self.progress_var.set(0)
        self.status_var.set("Siap")

    # === Core Prediction Methods ===

    def perbarui_tampilan_data(self):
        """Memperbarui tampilan data di UI"""
        # Clear treeview
        for item in self.pohon_data.get_children():
            self.pohon_data.delete(item)

        # Add data to treeview
        for i in range(len(self.data_historis['tahun'])):
            self.pohon_data.insert("", "end", values=(
                self.data_historis['tahun'][i],
                self.data_historis['inflasi'][i],
                self.data_historis['suku_bunga'][i],
                self.data_historis['indeks_usd'][i],
                self.data_historis['harga_emas'][i]
            ))

        # Update statistics
        self.update_statistics()

        # Update visualizations
        self.perbarui_visualisasi_data()

    def update_statistics(self):
        """Update statistik data"""
        if not self.data_historis['tahun']:
            return

        stats = []
        for key in ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']:
            data = np.array(self.data_historis[key])
            stats.append(f"{key.replace('_', ' ').title()}:")
            stats.append(f"  Mean: {np.mean(data):.2f}")
            stats.append(f"  Std: {np.std(data):.2f}")
            stats.append(f"  Min: {np.min(data):.2f}")
            stats.append(f"  Max: {np.max(data):.2f}")
            stats.append("")

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "\n".join(stats))
        self.stats_text.config(state=tk.DISABLED)

    def perbarui_visualisasi_data(self, event=None):
        """Memperbarui visualisasi data"""
        if not self.data_historis['tahun']:
            return

        self.gambar_data.clear()

        jenis_plot = self.jenis_plot.get()
        var_selected = self.var_visualisasi.get()

        if var_selected == "semua":
            # Create subplots
            axes = []
            axes.append(self.gambar_data.add_subplot(221))
            axes.append(self.gambar_data.add_subplot(222))
            axes.append(self.gambar_data.add_subplot(223))
            axes.append(self.gambar_data.add_subplot(224))

            tahun = self.data_historis['tahun']
            data_vars = ['harga_emas', 'inflasi', 'suku_bunga', 'indeks_usd']
            colors = ['gold', 'red', 'blue', 'green']
            titles = ['Harga Emas (USD)', 'Inflasi (%)', 'Suku Bunga (%)', 'Indeks USD']

            for i, (var, color, title) in enumerate(zip(data_vars, colors, titles)):
                data = self.data_historis[var]

                if jenis_plot == "line":
                    axes[i].plot(tahun, data, 'o-', color=color, linewidth=2, markersize=6)
                elif jenis_plot == "bar":
                    axes[i].bar(tahun, data, color=color, alpha=0.7)
                elif jenis_plot == "scatter":
                    axes[i].scatter(tahun, data, color=color, s=100)
                elif jenis_plot == "area":
                    axes[i].fill_between(tahun, data, alpha=0.7, color=color)

                axes[i].set_title(title, fontsize=10)
                axes[i].grid(True, linestyle='--', alpha=0.7)
                axes[i].tick_params(axis='x', rotation=45)

        else:
            # Single variable plot
            ax = self.gambar_data.add_subplot(111)
            tahun = self.data_historis['tahun']
            data = self.data_historis[var_selected]

            if jenis_plot == "line":
                ax.plot(tahun, data, 'o-', linewidth=3, markersize=8)
            elif jenis_plot == "bar":
                ax.bar(tahun, data, alpha=0.8)
            elif jenis_plot == "scatter":
                ax.scatter(tahun, data, s=150)
            elif jenis_plot == "area":
                ax.fill_between(tahun, data, alpha=0.7)

            ax.set_title(f'{var_selected.replace("_", " ").title()}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)

        self.gambar_data.tight_layout()
        self.kanvas_data.draw()

    def jalankan_prediksi_async(self):
        """Jalankan prediksi secara asynchronous"""
        if self.sedang_memproses:
            return

        def prediksi_thread():
            self.sedang_memproses = True
            try:
                self.jalankan_prediksi()
            finally:
                self.sedang_memproses = False
                self.reset_progress()

        thread = threading.Thread(target=prediksi_thread)
        thread.daemon = True
        thread.start()

    def jalankan_prediksi(self):
        """Menjalankan perhitungan prediksi dengan model yang dipilih"""
        try:
            self.update_progress(10, "Mempersiapkan data...")

            # Get parameters
            tahun = int(self.var_tahun.get())
            inflasi = float(self.var_inflasi.get())
            suku_bunga = float(self.var_suku_bunga.get())
            indeks_usd = float(self.var_usd.get())

            # Update prediction parameters
            self.parameter_prediksi.update({
                'tahun': tahun,
                'inflasi': inflasi,
                'suku_bunga': suku_bunga,
                'indeks_usd': indeks_usd
            })

            self.update_progress(30, "Mempersiapkan matriks...")

            # Prepare matrices
            M = np.array([
                self.data_historis['inflasi'],
                self.data_historis['suku_bunga'],
                self.data_historis['indeks_usd']
            ]).T

            y = np.array(self.data_historis['harga_emas'])
            Mx = np.array([inflasi, suku_bunga, indeks_usd])

            self.update_progress(50, "Menjalankan model...")

            # Get selected model
            model_type = self.model_terpilih.get()

            if model_type == "svd":
                harga_prediksi = self.prediksi_svd(M, y, Mx)
            else:
                harga_prediksi = self.prediksi_ml_model(M, y, Mx, model_type)

            self.update_progress(80, "Menghitung hasil...")

            # Store results
            self.hasil['harga_prediksi'] = harga_prediksi

            # Calculate confidence interval if requested
            if self.var_confidence.get():
                self.hasil['confidence_interval'] = self.hitung_confidence_interval(M, y, Mx)

            # Calculate residuals for training data
            if self.var_residual_analysis.get():
                self.hitung_residuals(M, y, model_type)

            self.update_progress(90, "Memperbarui tampilan...")

            # Update UI
            self.label_hasil.config(text=f"${harga_prediksi:.2f} USD")
            self.update_detail_results()
            self.perbarui_visualisasi_prediksi()

            self.update_progress(100, "Selesai!")

            messagebox.showinfo("Prediksi Selesai",
                                f"Harga emas prediksi untuk tahun {tahun}: ${harga_prediksi:.2f} USD")

        except ValueError as e:
            messagebox.showerror("Kesalahan Input", f"Input tidak valid: {str(e)}")
        except Exception as e:
            messagebox.showerror("Kesalahan Perhitungan", f"Terjadi kesalahan: {str(e)}")

    def prediksi_svd(self, M, y, Mx):
        """Prediksi menggunakan SVD"""
        # SVD decomposition
        U, S, VT = np.linalg.svd(M, full_matrices=False)
        V = VT.T

        # Calculate U^T * y
        UT_y = U.T @ y

        # Calculate pseudo-inverse
        S_inv = np.zeros_like(S)
        for i in range(len(S)):
            if S[i] > 1e-10:  # Avoid division by very small numbers
                S_inv[i] = 1 / S[i]

        # Calculate omega
        omega = V @ (S_inv * UT_y)

        # Store SVD components
        self.hasil['komponen_svd'] = {'U': U, 'S': S, 'V': V}
        self.hasil['vektor_omega'] = omega

        # Update SVD display
        self.update_svd_display(U, S, V, omega)

        # Calculate training predictions for residuals
        y_train_pred = M @ omega
        self.hasil['y_train_pred'] = y_train_pred

        # Prediction
        return Mx @ omega

    def prediksi_ml_model(self, M, y, Mx, model_type):
        """Prediksi menggunakan model ML - FIXED"""
        try:
            # FIXED: Properly access the model from dictionary
            if model_type not in self.model_alternatif:
                raise ValueError(f"Model '{model_type}' tidak ditemukan")

            # Create new instance to avoid state issues
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'lasso':
                model = Lasso(alpha=1.0)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Fit the model
            model.fit(M, y)

            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.hasil['feature_importance'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                self.hasil['feature_importance'] = np.abs(model.coef_)

            # Calculate training predictions for residuals
            y_train_pred = model.predict(M)
            self.hasil['y_train_pred'] = y_train_pred

            # Make prediction
            prediction = model.predict(Mx.reshape(1, -1))[0]

            # Update model display
            self.update_model_display(model, model_type)

            return prediction

        except Exception as e:
            raise Exception(f"Error dalam model {model_type}: {str(e)}")

    def hitung_residuals(self, M, y, model_type):
        """Calculate residuals for training data"""
        if 'y_train_pred' in self.hasil and self.hasil['y_train_pred'] is not None:
            residuals = y - self.hasil['y_train_pred']
            self.hasil['residuals'] = residuals
        else:
            self.hasil['residuals'] = None

    def update_svd_display(self, U, S, V, omega):
        """Update SVD components display"""
        svd_info = f"""=== Komponen SVD ===

Matrix U (Left Singular Vectors):
{U}

Singular Values (S):
{S}

Matrix V (Right Singular Vectors):
{V}

Vektor Omega (Koefisien):
{omega}

=== Informasi Tambahan ===
Condition Number: {np.max(S) / np.min(S[S > 1e-10]):.2f}
Rank: {np.sum(S > 1e-10)}
"""

        self.teks_svd.config(state=tk.NORMAL)
        self.teks_svd.delete(1.0, tk.END)
        self.teks_svd.insert(tk.END, svd_info)
        self.teks_svd.config(state=tk.DISABLED)

    def update_model_display(self, model, model_type):
        """Update ML model display"""
        model_info = f"=== Model: {model_type.upper()} ===\n\n"

        if hasattr(model, 'coef_'):
            model_info += f"Koefisien:\n{model.coef_}\n\n"

        if hasattr(model, 'intercept_'):
            model_info += f"Intercept: {model.intercept_:.4f}\n\n"

        if hasattr(model, 'feature_importances_'):
            model_info += f"Feature Importances:\n{model.feature_importances_}\n\n"

        # Calculate R² score for training data
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T
        y = np.array(self.data_historis['harga_emas'])

        y_pred_train = model.predict(M)
        r2_train = r2_score(y, y_pred_train)
        model_info += f"R² Score (Training): {r2_train:.4f}\n"

        self.teks_svd.config(state=tk.NORMAL)
        self.teks_svd.delete(1.0, tk.END)
        self.teks_svd.insert(tk.END, model_info)
        self.teks_svd.config(state=tk.DISABLED)

    def hitung_confidence_interval(self, M, y, Mx, confidence=0.95):
        """Hitung confidence interval untuk prediksi"""
        # Simplified bootstrap approach
        n_bootstrap = 100
        predictions = []

        for _ in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(len(y), size=len(y), replace=True)
            M_boot = M[indices]
            y_boot = y[indices]

            # Make prediction
            try:
                if self.model_terpilih.get() == "svd":
                    pred = self.prediksi_svd(M_boot, y_boot, Mx)
                else:
                    pred = self.prediksi_ml_model(M_boot, y_boot, Mx, self.model_terpilih.get())
                predictions.append(pred)
            except:
                continue

        if predictions:
            alpha = 1 - confidence
            lower = np.percentile(predictions, 100 * alpha / 2)
            upper = np.percentile(predictions, 100 * (1 - alpha / 2))
            return (lower, upper)

        return None

    def update_detail_results(self):
        """Update detail results display"""
        self.teks_detail.config(state=tk.NORMAL)
        self.teks_detail.delete(1.0, tk.END)

        detail = f"""Hasil Prediksi Harga Emas

Tahun: {self.parameter_prediksi['tahun']}
Model: {self.model_terpilih.get().upper()}

Parameter Input:
- Inflasi: {self.parameter_prediksi['inflasi']}%
- Suku Bunga: {self.parameter_prediksi['suku_bunga']}%
- Indeks USD: {self.parameter_prediksi['indeks_usd']}

Harga Prediksi: ${self.hasil['harga_prediksi']:.4f} USD

"""

        if self.hasil['confidence_interval']:
            ci = self.hasil['confidence_interval']
            detail += f"""Confidence Interval (95%):
- Lower: ${ci[0]:.2f}
- Upper: ${ci[1]:.2f}

"""

        if self.hasil['feature_importance'] is not None:
            detail += "Feature Importance:\n"
            features = ['Inflasi', 'Suku Bunga', 'Indeks USD']
            for i, (feat, imp) in enumerate(zip(features, self.hasil['feature_importance'])):
                detail += f"- {feat}: {imp:.4f}\n"

        detail += f"\nWaktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        self.teks_detail.insert(tk.END, detail)
        self.teks_detail.config(state=tk.DISABLED)

    def perbarui_visualisasi_prediksi(self, event=None):
        """Update prediction visualization - FIXED: Added event parameter"""
        if self.hasil['harga_prediksi'] is None:
            return

        self.gambar_pred.clear()

        view_type = self.pred_view.get()

        if view_type == "trend":
            self.plot_trend_prediction()
        elif view_type == "confidence":
            self.plot_confidence_prediction()
        elif view_type == "residuals":
            self.plot_residuals()
        elif view_type == "comparison":
            self.plot_model_comparison()

        self.gambar_pred.tight_layout()
        self.kanvas_pred.draw()

    def plot_trend_prediction(self):
        """Plot trend with prediction"""
        ax = self.gambar_pred.add_subplot(111)

        # Historical data
        tahun_hist = self.data_historis['tahun']
        harga_hist = self.data_historis['harga_emas']

        # Plot historical
        ax.plot(tahun_hist, harga_hist, 'o-', color='blue', linewidth=2,
                label='Data Historis', markersize=8)

        # Plot prediction
        tahun_pred = self.parameter_prediksi['tahun']
        harga_pred = self.hasil['harga_prediksi']

        ax.plot([tahun_hist[-1], tahun_pred], [harga_hist[-1], harga_pred],
                'o--', color='red', linewidth=2, label='Prediksi', markersize=8)

        # Highlight prediction point
        ax.scatter([tahun_pred], [harga_pred], color='red', s=150, zorder=5)

        # Add annotation
        ax.annotate(f"${harga_pred:.2f}",
                    xy=(tahun_pred, harga_pred),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

        ax.set_title(f'Prediksi Harga Emas - {tahun_pred}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Harga Emas (USD)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    def plot_confidence_prediction(self):
        """Plot prediction with confidence interval"""
        if not self.hasil['confidence_interval']:
            self.plot_trend_prediction()
            return

        ax = self.gambar_pred.add_subplot(111)

        # Historical data
        tahun_hist = self.data_historis['tahun']
        harga_hist = self.data_historis['harga_emas']

        ax.plot(tahun_hist, harga_hist, 'o-', color='blue', linewidth=2,
                label='Data Historis')

        # Prediction with confidence interval
        tahun_pred = self.parameter_prediksi['tahun']
        harga_pred = self.hasil['harga_prediksi']
        ci_lower, ci_upper = self.hasil['confidence_interval']

        ax.errorbar([tahun_pred], [harga_pred],
                    yerr=[[harga_pred - ci_lower], [ci_upper - harga_pred]],
                    fmt='ro', capsize=10, capthick=2, markersize=10,
                    label=f'Prediksi (CI 95%)')

        ax.set_title('Prediksi dengan Confidence Interval', fontsize=14)
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Harga Emas (USD)')
        ax.grid(True, alpha=0.7)
        ax.legend()

    def plot_residuals(self):
        """Plot residuals analysis - IMPLEMENTED"""
        if self.hasil['residuals'] is None:
            ax = self.gambar_pred.add_subplot(111)
            ax.text(0.5, 0.5, 'Residuals not available\nRun prediction first',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Residuals Analysis')
            return

        # Create 2x2 subplot layout for residuals analysis
        ax1 = self.gambar_pred.add_subplot(221)
        ax2 = self.gambar_pred.add_subplot(222)
        ax3 = self.gambar_pred.add_subplot(223)
        ax4 = self.gambar_pred.add_subplot(224)

        residuals = self.hasil['residuals']
        y_actual = np.array(self.data_historis['harga_emas'])
        y_pred = self.hasil['y_train_pred']
        tahun = self.data_historis['tahun']

        # 1. Residuals vs Fitted
        ax1.scatter(y_pred, residuals, alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)

        # 2. Normal Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True, alpha=0.3)

        # 3. Residuals vs Time (Years)
        ax3.plot(tahun, residuals, 'o-', alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Time')
        ax3.grid(True, alpha=0.3)

        # 4. Histogram of residuals
        ax4.hist(residuals, bins=5, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Histogram of Residuals')
        ax4.grid(True, alpha=0.3)

        # Add statistics text
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax4.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_model_comparison(self):
        """Plot comparison of different models - IMPLEMENTED"""
        # Run all models for comparison
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T
        y = np.array(self.data_historis['harga_emas'])
        Mx = np.array([
            self.parameter_prediksi['inflasi'],
            self.parameter_prediksi['suku_bunga'],
            self.parameter_prediksi['indeks_usd']
        ])

        models = ['svd', 'linear', 'ridge', 'lasso', 'random_forest']
        model_names = ['SVD', 'Linear', 'Ridge', 'Lasso', 'Random Forest']
        predictions = []
        r2_scores = []
        rmse_scores = []

        for model_type in models:
            try:
                if model_type == 'svd':
                    # SVD prediction
                    U, S, VT = np.linalg.svd(M, full_matrices=False)
                    V = VT.T
                    UT_y = U.T @ y
                    S_inv = np.zeros_like(S)
                    for i in range(len(S)):
                        if S[i] > 1e-10:
                            S_inv[i] = 1 / S[i]
                    omega = V @ (S_inv * UT_y)
                    pred = Mx @ omega
                    y_train_pred = M @ omega
                else:
                    # ML models
                    if model_type == 'linear':
                        model = LinearRegression()
                    elif model_type == 'ridge':
                        model = Ridge(alpha=1.0)
                    elif model_type == 'lasso':
                        model = Lasso(alpha=1.0)
                    elif model_type == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)

                    model.fit(M, y)
                    pred = model.predict(Mx.reshape(1, -1))[0]
                    y_train_pred = model.predict(M)

                predictions.append(pred)
                r2_scores.append(r2_score(y, y_train_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y, y_train_pred)))

            except Exception as e:
                predictions.append(0)
                r2_scores.append(0)
                rmse_scores.append(0)

        # Create comparison plots
        ax1 = self.gambar_pred.add_subplot(221)
        ax2 = self.gambar_pred.add_subplot(222)
        ax3 = self.gambar_pred.add_subplot(223)
        ax4 = self.gambar_pred.add_subplot(224)

        # 1. Predictions comparison
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        bars1 = ax1.bar(model_names, predictions, color=colors, alpha=0.7)
        ax1.set_title('Model Predictions Comparison')
        ax1.set_ylabel('Predicted Price (USD)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, pred in zip(bars1, predictions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'${pred:.0f}', ha='center', va='bottom', fontsize=8)

        # 2. R² scores comparison
        bars2 = ax2.bar(model_names, r2_scores, color=colors, alpha=0.7)
        ax2.set_title('R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Add value labels
        for bar, r2 in zip(bars2, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. RMSE comparison
        bars3 = ax3.bar(model_names, rmse_scores, color=colors, alpha=0.7)
        ax3.set_title('RMSE Comparison')
        ax3.set_ylabel('RMSE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar, rmse in zip(bars3, rmse_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{rmse:.1f}', ha='center', va='bottom', fontsize=8)

        # 4. Summary table
        ax4.axis('tight')
        ax4.axis('off')

        table_data = []
        for i, name in enumerate(model_names):
            table_data.append([name, f'${predictions[i]:.0f}', f'{r2_scores[i]:.3f}', f'{rmse_scores[i]:.1f}'])

        table = ax4.table(cellText=table_data,
                          colLabels=['Model', 'Prediction', 'R²', 'RMSE'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.25, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Model Performance Summary')

    # === Placeholder methods - implement as needed ===
    def simpan_pengaturan(self):
        """Simpan pengaturan aplikasi"""
        try:
            with open('settings.json', 'w') as f:
                json.dump(self.pengaturan, f, indent=2)
        except:
            pass

    def muat_pengaturan(self):
        """Muat pengaturan aplikasi"""
        try:
            with open('settings.json', 'r') as f:
                self.pengaturan.update(json.load(f))
        except:
            pass

    def sort_treeview(self, col):
        """Sort treeview by column"""
        pass

    def tambah_baris_data(self):
        """Add new data row"""
        pass

    def edit_baris_data(self):
        """Edit selected data row"""
        pass

    def hapus_baris_data(self):
        """Delete selected data row"""
        pass

    def impor_data(self):
        """Import data from file"""
        pass

    def ekspor_data(self):
        """Export data to file"""
        pass

    def ekspor_hasil(self):
        """Export results"""
        pass

    def refresh_data(self):
        """Refresh data display"""
        self.perbarui_tampilan_data()

    def reset_cache(self):
        """Reset application cache"""
        self.cache_perhitungan.clear()
        self.cache_visualisasi.clear()

    def buka_pengaturan(self):
        """Open settings dialog"""
        pass

    def analisis_komprehensif(self):
        """Run comprehensive analysis"""
        pass

    def jalankan_backtesting(self):
        """Run backtesting"""
        pass

    def jalankan_cross_validation(self):
        """Run cross validation"""
        pass

    def buka_grafik_interaktif(self):
        """Open interactive charts"""
        pass

    def buka_surface_plot_3d(self):
        """Open 3D surface plot"""
        pass

    def buka_heatmap_korelasi(self):
        """Open correlation heatmap"""
        pass

    def ubah_tema(self, tema):
        """Change application theme"""
        self.pengaturan['tema'] = tema

    def toggle_tema(self):
        """Toggle between light and dark theme"""
        current = self.pengaturan['tema']
        new_tema = 'gelap' if current == 'terang' else 'terang'
        self.ubah_tema(new_tema)

    def buka_panduan(self):
        """Open user guide"""
        messagebox.showinfo("Panduan", "Panduan penggunaan akan dibuka di browser.")

    def buka_tentang(self):
        """Show about dialog"""
        messagebox.showinfo("Tentang",
                            "Aplikasi Prediksi Harga Emas v2.0\n" +
                            "Menggunakan SVD dan Machine Learning\n" +
                            "© 2024")

    def simpan_grafik_data(self):
        """Save data chart"""
        pass

    def reset_zoom_data(self):
        """Reset data chart zoom"""
        pass

    def ekspor_laporan_pdf(self):
        """Export PDF report"""
        pass

    def ekspor_grafik(self):
        """Export charts"""
        pass


if __name__ == "__main__":
    root = tk.Tk()
    aplikasi = EnhancedAplikasiPrediksiHargaEmas(root)
    root.mainloop()