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

        # Model tambahan
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
        detail_scroll = ttk.Scrollbar(tab_detail, orient=tk.VERTICAL, comman