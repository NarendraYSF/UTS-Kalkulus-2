"""
===============================================================================
                            Dokumentasi Kode
===============================================================================

Judul: Aplikasi Rekomendasi E-Commerce dengan Sistem SVD yang Ditingkatkan
Deskripsi:
    Skrip ini mengimplementasikan sistem rekomendasi berbasis Singular Value Decomposition (SVD)
    untuk aplikasi e-commerce. Aplikasi ini menyediakan antarmuka pengguna grafis (GUI) interaktif
    yang memungkinkan pengguna untuk memuat data, melatih model, mengoptimalkan hyperparameter,
    dan mengevaluasi kinerja model dengan berbagai metrik. Aplikasi ini juga memungkinkan pengguna
    untuk membuat rekomendasi berdasarkan pengguna atau item serta visualisasi performa model.

    Fitur utama:
    - Generasi data sintetis untuk pengujian
    - Pelatihan model menggunakan berbagai algoritma SVD
    - Optimasi hyperparameter untuk meningkatkan akurasi model
    - Evaluasi dengan metrik RMSE, MAE, dan waktu eksekusi
    - Visualisasi distribusi rating dan performa model
    - Pengaturan tema antarmuka pengguna (light/dark mode)

Penulis: Narendra Yusuf 未来
Tanggal: May 23 2025
Versi: 2.0

===============================================================================
                            Deskripsi Data
===============================================================================

Data input terdiri dari rating yang diberikan oleh pengguna terhadap item di platform e-commerce:
    - 'user_id': Identitas unik untuk setiap pengguna
    - 'item_id': Identitas unik untuk setiap item
    - 'rating': Penilaian yang diberikan oleh pengguna terhadap item tersebut

Selain itu, aplikasi juga memungkinkan penggunaan data sintetis yang dapat dihasilkan melalui antarmuka.

===============================================================================
                            Ikhtisar Fungsionalitas
===============================================================================

1. **Pembuatan Data Sintetis**:
    - Pengguna dapat menghasilkan data rating sintetis dengan menentukan jumlah pengguna, item,
      dan tingkat kelangkaan data. Data ini digunakan untuk pelatihan dan pengujian model rekomendasi.

2. **Pelatihan Model SVD**:
    - Pengguna dapat memilih dari berbagai algoritma SVD yang tersedia untuk melatih model rekomendasi
      berdasarkan data yang dimiliki, termasuk penggunaan teknik SVD++ dan NMF.

3. **Optimasi Hyperparameter**:
    - Aplikasi mendukung optimasi hyperparameter untuk model rekomendasi dengan mengatur jumlah epoch,
      laju pembelajaran, dan regulasi untuk meningkatkan kinerja model.

4. **Evaluasi Model**:
    - Evaluasi model dilakukan dengan menghitung metrik seperti RMSE dan MAE, serta waktu eksekusi pelatihan
      dan pengujian. Metrik evaluasi lainnya, termasuk precision, recall, dan F1-score, juga dapat dihitung.

5. **Rekomendasi Berdasarkan Pengguna dan Item**:
    - Pengguna dapat memilih model yang telah dilatih dan menghasilkan rekomendasi untuk pengguna tertentu atau
      menemukan item yang serupa dengan item yang dipilih.

6. **Visualisasi**:
    - Hasil evaluasi dan performa model divisualisasikan dengan grafik distribusi rating, waktu eksekusi, dan
      perbandingan metrik antar algoritma.

7. **Tema Antarmuka Pengguna**:
    - Pengguna dapat memilih tema antarmuka gelap atau terang untuk pengalaman penggunaan yang lebih baik.

===============================================================================
                            Pembagian Kode
===============================================================================

1. **Kelas EnhancedSVDRecommenderApp**:
    - Kelas utama yang mengelola antarmuka pengguna (GUI) dan fungsionalitas aplikasi. Kelas ini mencakup
      metode untuk memuat data, melatih model, mengoptimalkan parameter, serta mengevaluasi dan
      memvisualisasikan hasil.

2. **Fungsi Setup UI**:
    - Fungsi-fungsi ini mendefinisikan struktur GUI, termasuk tab untuk data, algoritma, optimasi,
      rekomendasi, evaluasi, visualisasi, dan pengaturan. Fungsi-fungsi ini menangani pembuatan tombol,
      label, input, dan tabel.

3. **Fungsi Pembuatan Data**:
    - Fungsi untuk menghasilkan data sintetis berdasarkan parameter yang ditentukan oleh pengguna,
      seperti jumlah pengguna, item, dan distribusi rating.

4. **Fungsi Optimasi**:
    - Fungsi untuk melakukan optimasi hyperparameter untuk model rekomendasi, menguji berbagai konfigurasi
      untuk meningkatkan kinerja model.

5. **Fungsi Evaluasi Model**:
    - Fungsi untuk mengevaluasi model menggunakan metrik akurasi dan performa lainnya. Ini mencakup penghitungan
      RMSE, MAE, dan waktu pelatihan serta pengujian, serta visualisasi hasil perbandingan.

6. **Fungsi Rekomendasi**:
    - Fungsi untuk menghasilkan rekomendasi item berdasarkan model yang telah dilatih, serta untuk menemukan
      item yang mirip dengan item tertentu.

7. **Fungsi Visualisasi**:
    - Fungsi untuk menampilkan grafik distribusi rating, performa waktu pelatihan dan pengujian, serta perbandingan
      antar algoritma dengan menggunakan berbagai jenis chart.

8. **Pengaturan Tema**:
    - Fungsi untuk mengganti antara mode gelap dan terang, serta untuk mengonfigurasi preferensi pengguna lainnya.

===============================================================================
                            Instruksi Penggunaan
===============================================================================

1. **Menjalankan Aplikasi**:
    - Jalankan aplikasi dan pilih dataset yang ingin digunakan untuk melatih model rekomendasi atau buat
      data sintetis.

2. **Menggunakan Tab Data**:
    - Pilih jumlah pengguna, item, dan sparsitas data untuk menghasilkan data sintetis atau impor data dari file
      CSV/Excel.

3. **Melatih Model**:
    - Pilih model rekomendasi yang diinginkan dari tab 'Algorithms' dan latih model berdasarkan data yang tersedia.

4. **Optimasi dan Evaluasi**:
    - Gunakan tab 'Optimization' untuk mengoptimalkan hyperparameter model. Lakukan evaluasi di tab 'Evaluation'
      untuk menghitung metrik akurasi dan performa.

5. **Rekomendasi dan Visualisasi**:
    - Gunakan tab 'Recommendations' untuk menghasilkan rekomendasi berdasarkan model yang dilatih. Gunakan tab
      'Visualization' untuk melihat grafik distribusi rating dan metrik performa model.

6. **Pengaturan Tema**:
    - Sesuaikan tema aplikasi antara mode gelap atau terang menggunakan pengaturan pada tab 'Settings'.

7. **Menyimpan dan Memuat Data**:
    - Gunakan opsi pada menu untuk menyimpan model, memuat model, atau mengekspor rekomendasi ke file CSV.

===============================================================================
"""
#region
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, SVDpp, SlopeOne, NMF, CoClustering, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import warnings
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, colorchooser
from tkinter.font import Font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib.figure import Figure
import os
import pickle
import datetime
import webbrowser
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import io
from PIL import Image, ImageTk
import random
import json


matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')
#endregion

class EnhancedSVDRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced E-Commerce Recommendation System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        self.root.minsize(1100, 700)

        # Define styles and colors
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#e8eef1"
        self.accent_color = "#5cb85c"
        self.text_color = "#333333"
        self.error_color = "#d9534f"
        self.warning_color = "#f0ad4e"
        self.success_color = "#5cb85c"
        self.is_dark_mode = False

        # Initialize variables
        self.df = None
        self.data = None
        self.model = None
        self.models = {}  # Dictionary to store multiple trained models
        self.results = {}
        self.best_params = {}
        self.trainset = None
        self.testset = None
        self.current_model_name = "SVD"  # Default model
        self.selected_algorithms = []
        self.precision_recall_data = {}
        self.evaluation_metrics = {}
        self.last_saved_path = None
        self.imported_data_path = None
        self.recent_files = []
        self.max_recent_files = 5
        self.active_threads = []
        self.item_features = None  # For content-based recommendations
        self.config = self.load_config()

        # Load recent files from config
        if 'recent_files' in self.config:
            self.recent_files = self.config['recent_files']

        # Load color theme from config
        if 'dark_mode' in self.config:
            self.is_dark_mode = self.config['dark_mode']

        # Setup styles
        self.setup_styles()

        # Create menu
        self.create_menu()

        # Create header
        self.create_header()

        # Create tabs
        self.create_notebook()

        # Create status bar
        self.create_status_bar()

        # Initialize UI
        self.setup_data_tab()
        self.setup_algorithm_tab()
        self.setup_optimization_tab()
        self.setup_recommendation_tab()
        self.setup_evaluation_tab()
        self.setup_visualization_tab()
        self.setup_settings_tab()

        # Set default status
        self.update_status("Ready. Start by generating or loading data in the Data tab.")

        # Apply theme based on config
        if self.is_dark_mode:
            self.apply_dark_theme()

    def debug_surprise_imports(self):
        """Debug function to verify Surprise imports at runtime"""
        try:
            from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans
            # Test instantiation
            svd_test = SVD()
            svdpp_test = SVDpp()
            nmf_test = NMF()
            knn_basic_test = KNNBasic()
            knn_means_test = KNNWithMeans()
            print("✓ Surprise algorithms successfully imported and instantiated")

            # Return the classes to ensure they're accessible
            return {"SVD": SVD, "SVDpp": SVDpp, "NMF": NMF,
                    "KNNBasic": KNNBasic, "KNNWithMeans": KNNWithMeans}
        except Exception as e:
            print(f"❌ Error in surprise imports: {e}")
            return None

    def load_config(self):
        """Load configuration from file"""
        config_path = os.path.join(os.path.expanduser("~"), ".svd_recommender_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        return {}

    def save_config(self):
        """Save configuration to file"""
        config = {
            'recent_files': self.recent_files,
            'dark_mode': self.is_dark_mode
        }
        config_path = os.path.join(os.path.expanduser("~"), ".svd_recommender_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update_recent_files(self, file_path):
        """Update list of recent files"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]

        # Update menu
        self.update_recent_files_menu()

        # Save config
        self.save_config()

    def create_header(self):
        """Create application header with logo and title"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # Try to create a logo (you can replace this with your own logo if available)
        try:
            # Create a simple logo using matplotlib
            logo_fig = Figure(figsize=(1, 1), dpi=100)
            logo_ax = logo_fig.add_subplot(111)

            # Draw a simple SVD-like logo
            u = np.linspace(0, 2 * np.pi, 100)
            x = 16 * np.sin(u) ** 3
            y = 13 * np.cos(u) - 5 * np.cos(2 * u) - 2 * np.cos(3 * u) - np.cos(4 * u)
            logo_ax.plot(x, y, color=self.primary_color)
            logo_ax.axis('off')
            logo_ax.set_aspect('equal')
            logo_fig.tight_layout(pad=0)

            # Convert to PhotoImage
            buf = io.BytesIO()
            logo_fig.savefig(buf, format='png', transparent=True)
            buf.seek(0)
            logo_image = Image.open(buf)
            logo_image = logo_image.resize((60, 60), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)

            # Create and place logo label
            logo_label = ttk.Label(header_frame, image=logo_photo, background=self.secondary_color)
            logo_label.image = logo_photo  # Keep a reference to prevent garbage collection
            logo_label.pack(side=tk.LEFT, padx=(0, 10))
        except Exception as e:
            print(f"Could not create logo: {e}")

        # Title and subtitle
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)

        title_label = ttk.Label(title_frame, text="Enhanced E-Commerce Recommendation System",
                                font=("Helvetica", 16, "bold"), foreground=self.primary_color)
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(title_frame, text="Build, optimize, and evaluate recommendation models",
                                   font=("Helvetica", 10), foreground=self.text_color)
        subtitle_label.pack(anchor=tk.W)

    def create_notebook(self):
        """Create tabbed notebook for different sections"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.algorithm_tab = ttk.Frame(self.notebook)
        self.optimization_tab = ttk.Frame(self.notebook)
        self.recommendation_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        # Add tabs to notebook
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.algorithm_tab, text="Algorithms")
        self.notebook.add(self.optimization_tab, text="Optimization")
        self.notebook.add(self.recommendation_tab, text="Recommendations")
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.settings_tab, text="Settings")

    def create_status_bar(self):
        """Create status bar with progress indicator"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        # Status label
        self.status_label = ttk.Label(status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=100)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))

    def update_status(self, message, start_progress=False):
        """Update status bar message and progress indicator"""
        self.status_label.config(text=message)

        if start_progress:
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()

    def setup_data_tab(self):
        """Setup the Data tab UI"""
        # Create left and right frames
        left_frame = ttk.Frame(self.data_tab, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # OPTION 1: Remove this label that's using grid() since it seems redundant
        # (you already have "Number of Users:" in the gen_frame)
        # label = ttk.Label(left_frame, text="Number of Users:", background=self.secondary_color, foreground=self.text_color)
        # label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # OR OPTION 2: Change it to use pack() like the other widgets in left_frame
        label = ttk.Label(left_frame, text="Number of Users:", background=self.secondary_color,
                          foreground=self.text_color)
        label.pack(anchor=tk.W, padx=5, pady=5)  # Changed from grid() to pack()

        # Left frame - Data Generation
        gen_frame = ttk.LabelFrame(left_frame, text="Generate Synthetic Data")
        gen_frame.pack(fill=tk.X, pady=(0, 10))

        # Rest of your code remains the same...

        # Parameters for data generation
        ttk.Label(gen_frame, text="Number of Users:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_users_var = tk.IntVar(value=500)
        ttk.Spinbox(gen_frame, from_=100, to=10000, textvariable=self.num_users_var, width=10).grid(row=0, column=1,
                                                                                                    padx=5, pady=5)

        ttk.Label(gen_frame, text="Number of Items:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_items_var = tk.IntVar(value=100)
        ttk.Spinbox(gen_frame, from_=50, to=5000, textvariable=self.num_items_var, width=10).grid(row=1, column=1,
                                                                                                  padx=5, pady=5)

        ttk.Label(gen_frame, text="Sparsity:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.sparsity_var = tk.DoubleVar(value=0.9)
        ttk.Spinbox(gen_frame, from_=0.5, to=0.99, increment=0.01, textvariable=self.sparsity_var, width=10).grid(row=2,
                                                                                                                  column=1,
                                                                                                                  padx=5,
                                                                                                                  pady=5)

        ttk.Label(gen_frame, text="Rating Distribution:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.rating_dist_var = tk.StringVar(value="skewed")
        ttk.Combobox(gen_frame, textvariable=self.rating_dist_var, values=["skewed", "normal", "uniform"],
                     width=10).grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(gen_frame, text="Random Seed:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(gen_frame, from_=1, to=1000, textvariable=self.seed_var, width=10).grid(row=4, column=1, padx=5,
                                                                                            pady=5)

        # Generate button
        ttk.Button(gen_frame, text="Generate Data", command=self.generate_data).grid(row=5, column=0, columnspan=2,
                                                                                     pady=10)

        # Import/Export frame
        io_frame = ttk.LabelFrame(left_frame, text="Import/Export")
        io_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(io_frame, text="Import Data", command=self.import_data).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(io_frame, text="Export Data", command=self.export_data).pack(fill=tk.X, padx=5, pady=5)

        # Statistics frame
        stats_frame = ttk.LabelFrame(left_frame, text="Data Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, width=40, height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.stats_text.config(state=tk.DISABLED)

        # Right frame - Data Visualization
        viz_frame = ttk.LabelFrame(right_frame, text="Data Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.data_fig_frame = ttk.Frame(viz_frame)
        self.data_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial message
        ttk.Label(self.data_fig_frame, text="Generate data to view distribution chart",
                  font=("Helvetica", 11)).pack(expand=True)

    def setup_algorithm_tab(self):
        """Setup the Algorithms tab UI"""
        # Create left and right frames
        left_frame = ttk.Frame(self.algorithm_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(self.algorithm_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame - Algorithm Selection
        alg_frame = ttk.LabelFrame(left_frame, text="Select Algorithms to Compare")
        alg_frame.pack(fill=tk.X, pady=(0, 10))

        # Checkboxes for algorithms
        self.alg_vars = {}
        algorithms = [
            ("SVD", "Singular Value Decomposition"),
            ("SVDpp", "SVD++"),
            ("NMF", "Non-negative Matrix Factorization"),
            ("SlopeOne", "Slope One"),
            ("KNNBasic", "K-Nearest Neighbors Basic"),
            ("KNNWithMeans", "KNN with Means"),
            ("KNNWithZScore", "KNN with Z-Score"),
            ("CoClustering", "Co-Clustering")
        ]

        for i, (algo, desc) in enumerate(algorithms):
            var = tk.BooleanVar(value=True if algo == "SVD" else False)
            self.alg_vars[algo] = var
            cb = ttk.Checkbutton(alg_frame, text=f"{algo} - {desc}", variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

        # Compare button
        ttk.Button(alg_frame, text="Compare Algorithms", command=self.compare_algorithms).grid(row=len(algorithms),
                                                                                               column=0, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(left_frame, text="Comparison Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for results
        columns = ("RMSE", "MAE", "Train Time", "Test Time")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)

        # Set column headings
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=80, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right frame - Visualization tabs
        viz_notebook = ttk.Notebook(right_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Create visualization frames
        self.rmse_chart_frame = ttk.Frame(viz_notebook)
        self.time_chart_frame = ttk.Frame(viz_notebook)
        self.metrics_chart_frame = ttk.Frame(viz_notebook)

        # Add frames to notebook
        viz_notebook.add(self.rmse_chart_frame, text="RMSE Comparison")
        viz_notebook.add(self.time_chart_frame, text="Time Performance")
        viz_notebook.add(self.metrics_chart_frame, text="Metrics Comparison")

        # Initial message
        self.algorithm_fig_frame = ttk.Frame(self.rmse_chart_frame)
        self.algorithm_fig_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.algorithm_fig_frame, text="Run comparison to view results chart",
                  font=("Helvetica", 11)).pack(expand=True)

    def setup_optimization_tab(self):
        """Setup the Optimization tab UI"""
        # Create left and right frames
        left_frame = ttk.Frame(self.optimization_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(self.optimization_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame - First optimization round
        first_frame = ttk.LabelFrame(left_frame, text="First Optimization Round")
        first_frame.pack(fill=tk.X, pady=(0, 10))

        # Model selection
        ttk.Label(first_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.opt_model_var = tk.StringVar(value="SVD")
        ttk.Combobox(first_frame, textvariable=self.opt_model_var,
                     values=["SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans"],
                     width=15).grid(row=0, column=1, padx=5, pady=5)

        # Parameter ranges
        ttk.Label(first_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="10, 20, 30")
        ttk.Entry(first_frame, textvariable=self.epochs_var, width=20).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(first_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="0.002, 0.005, 0.01")
        ttk.Entry(first_frame, textvariable=self.lr_var, width=20).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(first_frame, text="Regularization:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.reg_var = tk.StringVar(value="0.02, 0.05, 0.1")
        ttk.Entry(first_frame, textvariable=self.reg_var, width=20).grid(row=3, column=1, padx=5, pady=5)

        # Run button
        ttk.Button(first_frame, text="Run First Optimization", command=self.run_first_optimization).grid(row=4,
                                                                                                         column=0,
                                                                                                         columnspan=2,
                                                                                                         pady=10)

        # Second optimization round
        second_frame = ttk.LabelFrame(left_frame, text="Second Optimization Round")
        second_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(second_frame, text="Factors:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.factors_var = tk.StringVar(value="50, 100, 150, 200")
        ttk.Entry(second_frame, textvariable=self.factors_var, width=20).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(second_frame, text="Run Second Optimization", command=self.run_second_optimization).grid(row=1,
                                                                                                            column=0,
                                                                                                            columnspan=2,
                                                                                                            pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(left_frame, text="Optimization Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.opt_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=40, height=15)
        self.opt_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.opt_results_text.config(state=tk.DISABLED)

        # Right frame - Visualization
        viz_frame = ttk.LabelFrame(right_frame, text="Optimization Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True)

        self.opt_fig_frame = ttk.Frame(viz_frame)
        self.opt_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial message
        ttk.Label(self.opt_fig_frame, text="Run optimization to view results",
                  font=("Helvetica", 11)).pack(expand=True)

    def setup_recommendation_tab(self):
        """Setup the Recommendations tab UI"""
        # Create frames
        top_frame = ttk.Frame(self.recommendation_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        middle_frame = ttk.Frame(self.recommendation_tab)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        bottom_frame = ttk.Frame(self.recommendation_tab)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Top frame - Model training
        train_frame = ttk.LabelFrame(top_frame, text="Train Model")
        train_frame.pack(fill=tk.X, pady=(0, 10))

        # Model selection
        ttk.Label(train_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="SVD")
        ttk.Combobox(train_frame, textvariable=self.model_var,
                     values=["SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans", "SlopeOne", "CoClustering"],
                     width=15).grid(row=0, column=1, padx=5, pady=5)

        # Train button
        ttk.Button(train_frame, text="Train Model", command=self.train_final_model).grid(row=0, column=2, padx=5,
                                                                                         pady=5)

        # Model info
        self.model_info_label = ttk.Label(train_frame, text="No model trained yet")
        self.model_info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        # Middle frame - Get recommendations
        rec_frame = ttk.LabelFrame(middle_frame, text="Get Recommendations")
        rec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # User selection
        user_frame = ttk.Frame(rec_frame)
        user_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(user_frame, text="Select User:").pack(side=tk.LEFT, padx=(0, 5))
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(user_frame, textvariable=self.user_var, width=15)
        self.user_combo.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(user_frame, text="Get Recommendations", command=self.get_recommendations).pack(side=tk.LEFT)

        # Recommendations treeview
        rec_tree_frame = ttk.Frame(rec_frame)
        rec_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("Item ID", "Predicted Rating")
        self.rec_tree = ttk.Treeview(rec_tree_frame, columns=columns, show="headings", height=10)

        # Set column headings
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=100, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(rec_tree_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right frame - Similar items
        similar_frame = ttk.LabelFrame(middle_frame, text="Find Similar Items")
        similar_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Item selection
        item_frame = ttk.Frame(similar_frame)
        item_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(item_frame, text="Select Item:").pack(side=tk.LEFT, padx=(0, 5))
        self.item_var = tk.StringVar()
        self.item_combo = ttk.Combobox(item_frame, textvariable=self.item_var, width=15)
        self.item_combo.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(item_frame, text="Find Similar Items", command=self.find_similar_items).pack(side=tk.LEFT)

        # Similar items treeview
        similar_tree_frame = ttk.Frame(similar_frame)
        similar_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("Item ID", "Similarity Score")
        self.similar_tree = ttk.Treeview(similar_tree_frame, columns=columns, show="headings", height=10)

        # Set column headings
        for col in columns:
            self.similar_tree.heading(col, text=col)
            self.similar_tree.column(col, width=100, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(similar_tree_frame, orient=tk.VERTICAL, command=self.similar_tree.yview)
        self.similar_tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.similar_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bottom frame - Best/Worst predictions
        pred_notebook = ttk.Notebook(bottom_frame)
        pred_notebook.pack(fill=tk.BOTH, expand=True)

        # Best predictions frame
        best_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(best_frame, text="Best Predictions")

        # Best predictions treeview
        columns = ("User ID", "Item ID", "True Rating", "Predicted", "Error")
        self.best_tree = ttk.Treeview(best_frame, columns=columns, show="headings", height=8)

        # Set column headings
        for col in columns:
            self.best_tree.heading(col, text=col)
            self.best_tree.column(col, width=80, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(best_frame, orient=tk.VERTICAL, command=self.best_tree.yview)
        self.best_tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.best_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Worst predictions frame
        worst_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(worst_frame, text="Worst Predictions")

        # Worst predictions treeview
        columns = ("User ID", "Item ID", "True Rating", "Predicted", "Error")
        self.worst_tree = ttk.Treeview(worst_frame, columns=columns, show="headings", height=8)

        # Set column headings
        for col in columns:
            self.worst_tree.heading(col, text=col)
            self.worst_tree.column(col, width=80, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(worst_frame, orient=tk.VERTICAL, command=self.worst_tree.yview)
        self.worst_tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        self.worst_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_evaluation_tab(self):
        """Setup the Evaluation tab UI"""
        # Create left and right frames
        left_frame = ttk.Frame(self.evaluation_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(self.evaluation_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame - Evaluation parameters
        param_frame = ttk.LabelFrame(left_frame, text="Evaluation Parameters")
        param_frame.pack(fill=tk.X, pady=(0, 10))

        # Model selection
        ttk.Label(param_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.eval_model_var = tk.StringVar(value="SVD")
        ttk.Combobox(param_frame, textvariable=self.eval_model_var,
                     values=["SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans", "SlopeOne", "CoClustering"],
                     width=15).grid(row=0, column=1, padx=5, pady=5)

        # Cross-validation folds
        ttk.Label(param_frame, text="CV Folds:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.cv_folds_var = tk.IntVar(value=5)
        ttk.Spinbox(param_frame, from_=0, to=10, textvariable=self.cv_folds_var, width=10).grid(row=1, column=1, padx=5,
                                                                                                pady=5)
        ttk.Label(param_frame, text="(0 to skip CV)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

        # Test size
        ttk.Label(param_frame, text="Test Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(param_frame, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_size_var, width=10).grid(
            row=2, column=1, padx=5, pady=5)

        # Random seed
        ttk.Label(param_frame, text="Random Seed:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.eval_seed_var = tk.IntVar(value=42)
        ttk.Spinbox(param_frame, from_=1, to=1000, textvariable=self.eval_seed_var, width=10).grid(row=3, column=1,
                                                                                                   padx=5, pady=5)

        # K for precision/recall
        ttk.Label(param_frame, text="K for P/R:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.k_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=50, textvariable=self.k_var, width=10).grid(row=4, column=1, padx=5,
                                                                                         pady=5)

        # Metrics selection
        metrics_frame = ttk.LabelFrame(param_frame, text="Metrics")
        metrics_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W + tk.E, padx=5, pady=5)

        # Create metric checkboxes
        self.metric_vars = {}
        metrics = [
            ("precision", "Precision@K"),
            ("recall", "Recall@K"),
            ("f1", "F1@K"),
            ("coverage", "Coverage"),
            ("diversity", "Diversity"),
            ("novelty", "Novelty")
        ]

        for i, (metric, label) in enumerate(metrics):
            var = tk.BooleanVar(value=True if metric in ["precision", "recall"] else False)
            self.metric_vars[metric] = var
            cb = ttk.Checkbutton(metrics_frame, text=label, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky=tk.W, padx=5, pady=2)

        # Run button
        ttk.Button(param_frame, text="Run Evaluation", command=self.run_evaluation).grid(row=6, column=0, columnspan=3,
                                                                                         pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(left_frame, text="Evaluation Results")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.eval_results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=40, height=15)
        self.eval_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right frame - Visualization tabs
        viz_notebook = ttk.Notebook(right_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Create visualization frames
        self.roc_frame = ttk.Frame(viz_notebook)
        self.pr_frame = ttk.Frame(viz_notebook)
        self.cv_frame = ttk.Frame(viz_notebook)

        # Add frames to notebook
        viz_notebook.add(self.roc_frame, text="ROC Curve")
        viz_notebook.add(self.pr_frame, text="Precision-Recall")
        viz_notebook.add(self.cv_frame, text="Cross-Validation")

        # Initial messages
        for frame in [self.roc_frame, self.pr_frame, self.cv_frame]:
            ttk.Label(frame, text="Run evaluation to view results",
                      font=("Helvetica", 11)).pack(expand=True)

    def setup_visualization_tab(self):
        """Setup the Visualization tab UI"""
        # Create left and right frames
        left_frame = ttk.Frame(self.visualization_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = ttk.Frame(self.visualization_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame - Visualization controls
        control_frame = ttk.LabelFrame(left_frame, text="Visualization Controls")
        control_frame.pack(fill=tk.BOTH, expand=True)

        # Chart type
        ttk.Label(control_frame, text="Chart Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.chart_type_var = tk.StringVar(value="distribution")
        chart_types = [
            ("distribution", "Rating Distribution"),
            ("user_activity", "User Activity"),
            ("item_popularity", "Item Popularity"),
            ("heatmap", "Rating Heatmap"),
            ("user_similarity", "User Similarity"),
            ("item_similarity", "Item Similarity")
        ]
        chart_combo = ttk.Combobox(control_frame, textvariable=self.chart_type_var,
                                   values=[t[0] for t in chart_types], width=15)
        chart_combo.grid(row=0, column=1, padx=5, pady=5)

        # Sample size
        ttk.Label(control_frame, text="Sample Size:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_size_var = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=10, to=1000, textvariable=self.sample_size_var, width=10).grid(row=1, column=1,
                                                                                                        padx=5, pady=5)

        # Color scheme
        ttk.Label(control_frame, text="Color Scheme:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.color_scheme_var = tk.StringVar(value="viridis")
        color_schemes = ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds", "YlOrRd",
                         "coolwarm"]
        ttk.Combobox(control_frame, textvariable=self.color_scheme_var, values=color_schemes, width=15).grid(row=2,
                                                                                                             column=1,
                                                                                                             padx=5,
                                                                                                             pady=5)

        # Figure size
        ttk.Label(control_frame, text="Figure Width:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.fig_width_var = tk.IntVar(value=8)
        ttk.Spinbox(control_frame, from_=4, to=16, textvariable=self.fig_width_var, width=10).grid(row=3, column=1,
                                                                                                   padx=5, pady=5)

        ttk.Label(control_frame, text="Figure Height:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.fig_height_var = tk.IntVar(value=6)
        ttk.Spinbox(control_frame, from_=3, to=12, textvariable=self.fig_height_var, width=10).grid(row=4, column=1,
                                                                                                    padx=5, pady=5)

        # Generate button
        ttk.Button(control_frame, text="Generate Visualization", command=self.generate_visualization).grid(row=5,
                                                                                                           column=0,
                                                                                                           columnspan=2,
                                                                                                           pady=10)

        # Export button
        ttk.Button(control_frame, text="Export Chart", command=self.export_chart).grid(row=6, column=0, columnspan=2,
                                                                                       pady=5)

        # Right frame - Chart display
        chart_frame = ttk.LabelFrame(right_frame, text="Visualization")
        chart_frame.pack(fill=tk.BOTH, expand=True)

        self.chart_frame = ttk.Frame(chart_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial message
        ttk.Label(self.chart_frame, text="Select visualization type and click Generate",
                  font=("Helvetica", 11)).pack(expand=True)

    def setup_settings_tab(self):
        """Setup the Settings tab UI"""
        # Create main frame
        main_frame = ttk.Frame(self.settings_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Appearance settings
        appearance_frame = ttk.LabelFrame(main_frame, text="Appearance")
        appearance_frame.pack(fill=tk.X, pady=(0, 10))

        # Theme selection
        theme_frame = ttk.Frame(appearance_frame)
        theme_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=(0, 10))

        self.theme_var = tk.StringVar(value="Light" if not self.is_dark_mode else "Dark")
        ttk.Radiobutton(theme_frame, text="Light", variable=self.theme_var, value="Light",
                        command=lambda: self.apply_light_theme()).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(theme_frame, text="Dark", variable=self.theme_var, value="Dark",
                        command=lambda: self.apply_dark_theme()).pack(side=tk.LEFT)

        # Color customization
        color_frame = ttk.Frame(appearance_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(color_frame, text="Primary Color:").pack(side=tk.LEFT, padx=(0, 10))
        primary_color_btn = ttk.Button(color_frame, text="Choose", command=self.choose_primary_color)
        primary_color_btn.pack(side=tk.LEFT)

        # Performance settings
        perf_frame = ttk.LabelFrame(main_frame, text="Performance")
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        # Threading options
        thread_frame = ttk.Frame(perf_frame)
        thread_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(thread_frame, text="Parallel Processing:").pack(side=tk.LEFT, padx=(0, 10))

        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(thread_frame, text="Enable multi-threading", variable=self.parallel_var).pack(side=tk.LEFT)

        # Cache options
        cache_frame = ttk.Frame(perf_frame)
        cache_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(cache_frame, text="Cache Size:").pack(side=tk.LEFT, padx=(0, 10))

        self.cache_size_var = tk.IntVar(value=100)
        ttk.Spinbox(cache_frame, from_=0, to=1000, textvariable=self.cache_size_var, width=10).pack(side=tk.LEFT)
        ttk.Label(cache_frame, text="MB").pack(side=tk.LEFT, padx=(5, 0))

        # Clear cache button
        ttk.Button(cache_frame, text="Clear Cache", command=self.clear_cache).pack(side=tk.LEFT, padx=(20, 0))

        # Data settings
        data_frame = ttk.LabelFrame(main_frame, text="Data Settings")
        data_frame.pack(fill=tk.X, pady=(0, 10))

        # Default rating scale
        rating_frame = ttk.Frame(data_frame)
        rating_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(rating_frame, text="Default Rating Scale:").pack(side=tk.LEFT, padx=(0, 10))

        self.min_rating_var = tk.IntVar(value=1)
        ttk.Spinbox(rating_frame, from_=0, to=10, textvariable=self.min_rating_var, width=5).pack(side=tk.LEFT)

        ttk.Label(rating_frame, text="to").pack(side=tk.LEFT, padx=5)

        self.max_rating_var = tk.IntVar(value=5)
        ttk.Spinbox(rating_frame, from_=1, to=100, textvariable=self.max_rating_var, width=5).pack(side=tk.LEFT)

        # About section
        about_frame = ttk.LabelFrame(main_frame, text="About")
        about_frame.pack(fill=tk.BOTH, expand=True)

        about_text = """Enhanced E-Commerce Recommendation System

    Version 1.0

    A comprehensive tool for building and evaluating recommendation systems 
    for e-commerce applications.

    Features:
    • Multiple recommendation algorithms
    • Hyperparameter optimization
    • Comprehensive evaluation metrics
    • Interactive visualizations
    • User and item similarity analysis

    Developed with Python using Surprise, Pandas, NumPy, Matplotlib, and Tkinter.
    """

        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT)
        about_label.pack(padx=10, pady=10)

        # Buttons for documentation and about
        button_frame = ttk.Frame(about_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(button_frame, text="View Documentation", command=self.show_documentation).pack(side=tk.LEFT,
                                                                                                  padx=(0, 10))
        ttk.Button(button_frame, text="About", command=self.show_about).pack(side=tk.LEFT)

    def choose_primary_color(self):
        """Open color chooser dialog to select primary color"""
        color = colorchooser.askcolor(initialcolor=self.primary_color, title="Choose Primary Color")
        if color[1]:  # If a color was selected (not canceled)
            self.primary_color = color[1]
            # Update styles with new color
            self.style.configure('TButton', background=self.primary_color)
            self.style.map('TButton', background=[('active', self._adjust_color(self.primary_color, -20))])

    def _adjust_color(self, hex_color, amount):
        """Adjust color brightness by amount (-255 to 255)"""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        # Adjust color
        r = max(0, min(255, r + amount))
        g = max(0, min(255, g + amount))
        b = max(0, min(255, b + amount))

        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'

    def clear_cache(self):
        """Clear application cache"""
        # Implement cache clearing logic here
        messagebox.showinfo("Cache Cleared", "Application cache has been cleared.")

    def update_recent_files_menu(self):
        """Update the recent files menu"""
        if hasattr(self, 'recent_files_menu'):
            # Clear menu
            self.recent_files_menu.delete(0, tk.END)

            if not self.recent_files:
                self.recent_files_menu.add_command(label="No recent files", state=tk.DISABLED)
            else:
                for file_path in self.recent_files:
                    # Shorten path if too long
                    display_path = file_path
                    if len(display_path) > 50:
                        display_path = "..." + display_path[-47:]

                    self.recent_files_menu.add_command(
                        label=display_path,
                        command=lambda path=file_path: self.load_data_from_file(path)
                    )

                self.recent_files_menu.add_separator()
                self.recent_files_menu.add_command(label="Clear Recent Files", command=self.clear_recent_files)

    def clear_recent_files(self):
        """Clear the list of recent files"""
        self.recent_files = []
        self.update_recent_files_menu()
        self.save_config()

    def setup_styles(self):
        """Setup ttk styles for the application"""
        self.style = ttk.Style()
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.primary_color, foreground=self.text_color)
        self.style.configure('Treeview', background=self.secondary_color, foreground=self.text_color,
                             fieldbackground=self.secondary_color)
        self.style.configure('TNotebook', background=self.secondary_color)
        self.style.configure('TNotebook.Tab', background=self.secondary_color, foreground=self.text_color)

        self.style.map('TButton',
                       background=[('active', '#3d5c8c')],
                       foreground=[('active', 'white')])

        # Setup custom styles for tabs and notebook
        self.style.configure('TNotebook', background=self.secondary_color)
        self.style.configure('TNotebook.Tab', background=self.secondary_color, foreground=self.text_color)

        # Setup styles for treeview
        self.style.configure('Treeview',
                             background=self.secondary_color,
                             foreground=self.text_color,
                             rowheight=25,
                             fieldbackground=self.secondary_color)
        self.style.map('Treeview',
                       background=[('selected', self.primary_color)],
                       foreground=[('selected', 'white')])

    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        # Set dark mode colors
        self.primary_color = "#2c3e50"
        self.secondary_color = "#34495e"
        self.text_color = "#ecf0f1"
        self.is_dark_mode = True

        # Update styles
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.primary_color, foreground=self.text_color)
        self.style.configure('Treeview', background=self.secondary_color, foreground=self.text_color,
                             fieldbackground=self.secondary_color)

        # Update text widgets
        for widget in [self.stats_text, self.opt_results_text]:
            widget.config(bg=self.secondary_color, fg=self.text_color)

        # Update root and frames
        self.root.configure(bg=self.secondary_color)

        # Save config
        self.save_config()

    def apply_light_theme(self):
        """Apply light theme to the application"""
        # Set light mode colors
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#e8eef1"
        self.text_color = "#333333"
        self.is_dark_mode = False

        # Update styles
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.primary_color, foreground='white')

        # Update notebook styles
        self.style.configure('TNotebook', background=self.secondary_color)
        self.style.configure('TNotebook.Tab', background=self.secondary_color, foreground=self.text_color)

        # Update treeview styles
        self.style.configure('Treeview',
                             background='white',
                             foreground=self.text_color,
                             fieldbackground='white')

        # Update text widgets
        for widget in [self.stats_text, self.opt_results_text]:
            widget.config(bg='white', fg=self.text_color)

        # Update root and frames
        self.root.configure(bg=self.secondary_color)

        # Save config
        self.save_config()

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        if self.is_dark_mode:
            self.apply_light_theme()
        else:
            self.apply_dark_theme()

    def create_menu(self):
        """Create application menu"""
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Import Data", command=self.import_data)
        file_menu.add_command(label="Export Recommendations", command=self.export_recommendations)  # Updated line
        file_menu.add_separator()
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Load Model", command=self.load_model)

        # Recent files submenu
        self.recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_files_menu)
        self.update_recent_files_menu()

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)

        # Edit menu
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences", command=lambda: self.notebook.select(self.settings_tab))

        # View menu
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_theme)

        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)

    def new_project(self):
        """Start a new project by clearing all data"""
        if messagebox.askyesno("New Project",
                               "Are you sure you want to start a new project? All unsaved data will be lost."):
            self.df = None
            self.data = None
            self.model = None
            self.models = {}
            self.results = {}
            self.best_params = {}
            self.trainset = None
            self.testset = None
            self.current_model_name = "SVD"
            self.update_status("New project created. Generate or import data to begin.")

            # Reset UI elements
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.config(state=tk.DISABLED)

            self.opt_results_text.config(state=tk.NORMAL)
            self.opt_results_text.delete(1.0, tk.END)
            self.opt_results_text.config(state=tk.DISABLED)

            # Clear visualizations
            for widget in self.data_fig_frame.winfo_children():
                widget.destroy()

            for widget in self.algorithm_fig_frame.winfo_children():
                widget.destroy()

            # Reset trees
            for tree in [self.results_tree, self.rec_tree, self.best_tree, self.worst_tree]:
                for item in tree.get_children():
                    tree.delete(item)

            ttk.Label(self.data_fig_frame, text="Generate data to view distribution chart",
                      font=("Helvetica", 11)).pack(expand=True)

            ttk.Label(self.algorithm_fig_frame, text="Run comparison to view results chart",
                      font=("Helvetica", 11)).pack(expand=True)

    def import_data(self):
        """Import data from a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Import Data",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )

        if file_path:
            self.load_data_from_file(file_path)

    def load_data_from_file(self, file_path):
        """Load data from a file path"""
        self.update_status(f"Loading data from {file_path}...", start_progress=True)

        # Store the file path
        self.imported_data_path = file_path

        # Add to recent files
        self.update_recent_files(file_path)

        # Start loading in a thread
        threading.Thread(target=self._load_data_thread, args=(file_path,)).start()

    def _load_data_thread(self, file_path):
        """Background thread for loading data"""
        try:
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            else:
                # Try CSV as default
                self.df = pd.read_csv(file_path)

            # Check if the dataframe has the required columns
            required_columns = ['user_id', 'item_id', 'rating']
            missing_columns = [col for col in required_columns if col not in self.df.columns]

            if missing_columns:
                # Show column mapping dialog in main thread
                self.root.after(0, lambda: self._show_column_mapping_dialog(self.df.columns.tolist()))
                return

            # Continue with processing
            self._process_imported_data()

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Failed to load data: {err}"))

    def _show_column_mapping_dialog(self, available_columns):
        """Show dialog for mapping columns"""
        mapping_dialog = tk.Toplevel(self.root)
        mapping_dialog.title("Map Columns")
        mapping_dialog.geometry("400x300")
        mapping_dialog.grab_set()  # Modal dialog

        ttk.Label(mapping_dialog, text="Please map your columns to the required fields:").pack(pady=(10, 20))

        # User ID mapping
        user_frame = ttk.Frame(mapping_dialog)
        user_frame.pack(fill=tk.X, pady=5)
        ttk.Label(user_frame, text="User ID:").pack(side=tk.LEFT, padx=(10, 5))
        user_var = tk.StringVar()
        user_combo = ttk.Combobox(user_frame, textvariable=user_var, values=available_columns)
        user_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Item ID mapping
        item_frame = ttk.Frame(mapping_dialog)
        item_frame.pack(fill=tk.X, pady=5)
        ttk.Label(item_frame, text="Item ID:").pack(side=tk.LEFT, padx=(10, 5))
        item_var = tk.StringVar()
        item_combo = ttk.Combobox(item_frame, textvariable=item_var, values=available_columns)
        item_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Rating mapping
        rating_frame = ttk.Frame(mapping_dialog)
        rating_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rating_frame, text="Rating:").pack(side=tk.LEFT, padx=(10, 5))
        rating_var = tk.StringVar()
        rating_combo = ttk.Combobox(rating_frame, textvariable=rating_var, values=available_columns)
        rating_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Try to guess mappings
        for col in available_columns:
            col_lower = col.lower()
            if 'user' in col_lower:
                user_var.set(col)
            elif 'item' in col_lower or 'product' in col_lower:
                item_var.set(col)
            elif 'rating' in col_lower or 'score' in col_lower or 'rate' in col_lower:
                rating_var.set(col)

        # Buttons
        btn_frame = ttk.Frame(mapping_dialog)
        btn_frame.pack(fill=tk.X, pady=(20, 10))

        ttk.Button(btn_frame, text="Cancel", command=mapping_dialog.destroy).pack(side=tk.RIGHT, padx=(5, 10))

        def apply_mapping():
            # Rename columns based on mapping
            column_map = {
                user_var.get(): 'user_id',
                item_var.get(): 'item_id',
                rating_var.get(): 'rating'
            }

            # Check if all mappings are provided
            if not all(column_map.keys()):
                messagebox.showerror("Error", "Please select a column for each required field.")
                return

            try:
                # Rename columns
                self.df = self.df.rename(columns=column_map)
                # Close dialog
                mapping_dialog.destroy()
                # Continue processing
                self._process_imported_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply mapping: {str(e)}")

        ttk.Button(btn_frame, text="Apply", command=apply_mapping).pack(side=tk.RIGHT, padx=5)

    def _process_imported_data(self):
        """Process the imported data after loading"""
        try:
            # Convert rating to float if needed
            if self.df['rating'].dtype not in [np.float64, np.int64]:
                self.df['rating'] = self.df['rating'].astype(float)

            # Check for rating scale
            min_rating = self.df['rating'].min()
            max_rating = self.df['rating'].max()

            # Convert to Surprise format
            reader = Reader(rating_scale=(min_rating, max_rating))
            self.data = Dataset.load_from_df(self.df[['user_id', 'item_id', 'rating']], reader)

            # Update UI in main thread
            self.root.after(0, self._update_data_ui)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Failed to process data: {err}"))

    def _update_data_ui(self):
        """Update UI with generated data information"""
        # Update statistics text
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        # Calculate statistics
        total_ratings = len(self.df)
        unique_users = self.df['user_id'].nunique()
        unique_items = self.df['item_id'].nunique()
        rating_dist = self.df['rating'].value_counts(normalize=True).sort_index() * 100

        # Calculate density
        max_possible_ratings = unique_users * unique_items
        density = (total_ratings / max_possible_ratings) * 100

        # Additional statistics
        avg_rating = self.df['rating'].mean()
        median_rating = self.df['rating'].median()
        ratings_per_user = self.df.groupby('user_id').size()
        ratings_per_item = self.df.groupby('item_id').size()
        avg_ratings_per_user = ratings_per_user.mean()
        avg_ratings_per_item = ratings_per_item.mean()

        # Format statistics text
        stats = f"Total ratings: {total_ratings:,}\n"
        stats += f"Unique users: {unique_users:,}\n"
        stats += f"Unique items: {unique_items:,}\n"
        stats += f"Matrix density: {density:.2f}%\n\n"
        stats += f"Average rating: {avg_rating:.2f}\n"
        stats += f"Median rating: {median_rating:.1f}\n\n"
        stats += f"Avg ratings per user: {avg_ratings_per_user:.2f}\n"
        stats += f"Avg ratings per item: {avg_ratings_per_item:.2f}\n\n"
        stats += "Rating Distribution:\n"
        for rating, percentage in rating_dist.items():
            stats += f"Rating {rating}: {percentage:.2f}%\n"

        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)

        # Update visualization
        self._update_data_visualization()

        # Update dropdown values for user and item selection
        if hasattr(self, 'user_combo'):
            all_users = self.df['user_id'].unique().tolist()
            all_users.sort()
            self.user_combo['values'] = all_users[:100]  # Limit to first 100 for performance

        if hasattr(self, 'item_combo'):
            all_items = self.df['item_id'].unique().tolist()
            all_items.sort()
            self.item_combo['values'] = all_items[:100]  # Limit to first 100 for performance

        self.update_status(f"Data loaded: {total_ratings} ratings from {unique_users} users on {unique_items} items")

    def _update_data_visualization(self):
        """Update data visualization chart"""
        # Clear previous chart
        for widget in self.data_fig_frame.winfo_children():
            widget.destroy()

        # Create new figure
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Generate rating distribution chart
        rating_counts = self.df['rating'].value_counts().sort_index()
        bars = ax.bar(rating_counts.index, rating_counts.values, color=self.primary_color)

        # Add percentage labels on top of bars
        total = rating_counts.sum()
        for bar in bars:
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom')

        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating Distribution')
        ax.set_xticks(rating_counts.index)

        # Add chart to frame
        canvas = FigureCanvasTkAgg(fig, master=self.data_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store current figure
        self.current_figure = fig

    def compare_algorithms(self):
        """Compare different recommendation algorithms"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get selected algorithms
        selected_algorithms = []
        for class_name, var in self.alg_vars.items():
            if var.get():
                selected_algorithms.append(class_name)

        if not selected_algorithms:
            messagebox.showwarning("Warning", "Please select at least one algorithm to compare.")
            return

        self.update_status(f"Comparing {len(selected_algorithms)} algorithms...", start_progress=True)

        # Make sure we have fresh imports
        from surprise import SVD, SVDpp, SlopeOne, NMF, CoClustering, KNNBasic, KNNWithMeans, KNNWithZScore

        # Start comparison in a separate thread
        threading.Thread(target=self._compare_algorithms_thread,
                         args=(selected_algorithms,
                               {"SVD": SVD, "SVDpp": SVDpp, "NMF": NMF, "SlopeOne": SlopeOne,
                                "CoClustering": CoClustering, "KNNBasic": KNNBasic,
                                "KNNWithMeans": KNNWithMeans, "KNNWithZScore": KNNWithZScore})).start()

    def _compare_algorithms_thread(self, selected_algorithms, algorithm_classes):
        """Background thread for algorithm comparison"""
        try:
            # Split data into train and test sets
            trainset, testset = train_test_split(self.data, test_size=0.25, random_state=42)

            # Get fresh imports to be sure
            from surprise import SVD, SVDpp, SlopeOne, NMF, CoClustering, KNNBasic, KNNWithMeans, KNNWithZScore

            # Create algorithm instances with explicit constructor calls
            algorithms = {}
            for algo_name in selected_algorithms:
                if algo_name == "SVD":
                    algorithms[algo_name] = SVD()
                elif algo_name == "SVDpp":
                    algorithms[algo_name] = SVDpp()
                elif algo_name == "NMF":
                    algorithms[algo_name] = NMF()
                elif algo_name == "SlopeOne":
                    algorithms[algo_name] = SlopeOne()
                elif algo_name == "CoClustering":
                    algorithms[algo_name] = CoClustering()
                elif algo_name == "KNNBasic":
                    algorithms[algo_name] = KNNBasic()
                elif algo_name == "KNNWithMeans":
                    algorithms[algo_name] = KNNWithMeans()
                elif algo_name == "KNNWithZScore":
                    algorithms[algo_name] = KNNWithZScore()

            # Print debugging information
            for name, algo in algorithms.items():
                print(f"Created algorithm {name}: {type(algo)}")

            # Train and test each algorithm
            results = {}
            for name, algo in algorithms.items():
                print(f"Training algorithm {name}...")
                # Train
                start_time = datetime.datetime.now()
                algo.fit(trainset)
                train_time = (datetime.datetime.now() - start_time).total_seconds()
                print(f"Algorithm {name} trained successfully")

                # Test
                start_time = datetime.datetime.now()
                predictions = algo.test(testset)
                test_time = (datetime.datetime.now() - start_time).total_seconds()
                print(f"Algorithm {name} tested successfully")

                # Calculate metrics
                rmse = accuracy.rmse(predictions)
                mae = accuracy.mae(predictions)

                # Store results
                results[name] = {
                    'algorithm': algo,
                    'rmse': rmse,
                    'mae': mae,
                    'train_time': train_time,
                    'test_time': test_time,
                    'predictions': predictions
                }

                # Store model for later use
                self.models[name] = algo

            # Store results
            self.results = results

            # Update UI in main thread
            self.root.after(0, lambda: self._update_comparison_results(results))

        except Exception as e:
            error_msg = str(e)
            print(f"Error in algorithm comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Comparison failed: {err}"))

    def _update_comparison_results(self, results):
        """Update UI with algorithm comparison results"""
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Sort algorithms by RMSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])

        # Insert results into tree
        for name, result in sorted_results:
            values = (
                f"{result['rmse']:.4f}",
                f"{result['mae']:.4f}",
                f"{result['train_time']:.2f}",
                f"{result['test_time']:.2f}"
            )
            self.results_tree.insert('', 'end', text=name, values=values, tags=(name,))

        # Highlight best algorithm
        best_algo = sorted_results[0][0]
        self.results_tree.tag_configure(best_algo, background=self.accent_color)

        # Update visualization
        self._update_comparison_visualization(results)

        # Update status
        self.update_status(
            f"Comparison complete. Best algorithm: {best_algo} (RMSE: {sorted_results[0][1]['rmse']:.4f})")

    def _update_comparison_visualization(self, results):
        """Update visualization charts for algorithm comparison"""
        # Clear previous charts
        for widget in self.rmse_chart_frame.winfo_children():
            widget.destroy()

        for widget in self.time_chart_frame.winfo_children():
            widget.destroy()

        for widget in self.metrics_chart_frame.winfo_children():
            widget.destroy()

        # Get data for charts
        names = list(results.keys())
        rmse_values = [results[name]['rmse'] for name in names]
        mae_values = [results[name]['mae'] for name in names]
        train_times = [results[name]['train_time'] for name in names]
        test_times = [results[name]['test_time'] for name in names]

        # RMSE Chart
        rmse_fig = Figure(figsize=(8, 5))
        rmse_ax = rmse_fig.add_subplot(111)

        bars = rmse_ax.bar(names, rmse_values, color=self.primary_color)
        rmse_ax.set_ylabel('RMSE (lower is better)')
        rmse_ax.set_title('RMSE Comparison')
        rmse_ax.set_xticklabels(names, rotation=45, ha='right')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            rmse_ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        rmse_canvas = FigureCanvasTkAgg(rmse_fig, master=self.rmse_chart_frame)
        rmse_canvas.draw()
        rmse_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Time Comparison Chart
        time_fig = Figure(figsize=(8, 5))
        time_ax = time_fig.add_subplot(111)

        x = np.arange(len(names))
        width = 0.35

        train_bars = time_ax.bar(x - width / 2, train_times, width, label='Training Time')
        test_bars = time_ax.bar(x + width / 2, test_times, width, label='Testing Time')

        time_ax.set_ylabel('Time (seconds)')
        time_ax.set_title('Performance Time Comparison')
        time_ax.set_xticks(x)
        time_ax.set_xticklabels(names, rotation=45, ha='right')
        time_ax.legend()

        time_canvas = FigureCanvasTkAgg(time_fig, master=self.time_chart_frame)
        time_canvas.draw()
        time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Metrics Comparison
        metrics_fig = Figure(figsize=(8, 5))
        metrics_ax = metrics_fig.add_subplot(111)

        x = np.arange(len(names))
        width = 0.35

        rmse_bars = metrics_ax.bar(x - width / 2, rmse_values, width, label='RMSE')
        mae_bars = metrics_ax.bar(x + width / 2, mae_values, width, label='MAE')

        metrics_ax.set_ylabel('Error Value')
        metrics_ax.set_title('Error Metrics Comparison')
        metrics_ax.set_xticks(x)
        metrics_ax.set_xticklabels(names, rotation=45, ha='right')
        metrics_ax.legend()

        metrics_canvas = FigureCanvasTkAgg(metrics_fig, master=self.metrics_chart_frame)
        metrics_canvas.draw()
        metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Update the run_first_optimization method
    def run_first_optimization(self):
        """Run first round of hyperparameter optimization"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get selected model and parameters
        model_name = self.opt_model_var.get()

        # Parse parameter ranges
        try:
            epochs = [int(x.strip()) for x in self.epochs_var.get().split(',')]
            lr_values = [float(x.strip()) for x in self.lr_var.get().split(',')]
            reg_values = [float(x.strip()) for x in self.reg_var.get().split(',')]
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter comma-separated numbers.")
            return

        self.update_status(f"Running first optimization round for {model_name}...", start_progress=True)

        # Make sure we have fresh imports
        from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans

        # Start optimization in a separate thread
        # Thread call with models dictionary
        threading.Thread(target=self._run_first_optimization_thread,
                         args=(model_name, epochs, lr_values, reg_values,
                               {"SVD": SVD, "SVDpp": SVDpp, "NMF": NMF, "KNNBasic": KNNBasic,
                                "KNNWithMeans": KNNWithMeans})).start()

    def _run_first_optimization_thread(self, model_name, epochs, lr_values, reg_values, models):
        """Background thread for first optimization round"""
        try:
            # Get the model class from the dictionary
            algo_class = models[model_name]

            # Define parameter grid based on the model
            if model_name == "NMF":
                # For NMF, use separate regularization parameters instead of reg_all
                param_grid = {
                    'n_epochs': epochs,
                    'reg_pu': reg_values,  # regularization for user factors
                    'reg_qi': reg_values  # regularization for item factors
                    # Note: NMF doesn't use lr_all parameter
                }
            elif model_name in ["KNNBasic", "KNNWithMeans"]:
                # KNN models don't use these parameters
                param_grid = {}
                # For KNN models, we might want to optimize other parameters like k, sim_options, etc.
                self.root.after(0, lambda: messagebox.showinfo("Info",
                                                               f"{model_name} doesn't require hyperparameter optimization for the selected parameters."))
                return
            else:
                # For SVD and SVDpp, include all parameters
                param_grid = {
                    'n_epochs': epochs,
                    'lr_all': lr_values,
                    'reg_all': reg_values
                }

            # Create GridSearchCV
            gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=3)
            gs.fit(self.data)

            # Extract best params and scores
            best_params = gs.best_params['rmse']
            best_rmse = gs.best_score['rmse']
            best_mae = gs.best_score['mae']

            # Store results
            self.best_params['first_round'] = best_params
            self.current_model_name = model_name

            # Create results dict for visualization
            cv_results = gs.cv_results
            param_combos = []
            rmse_means = []
            mae_means = []

            for i, params in enumerate(cv_results['params']):
                if model_name == "NMF":
                    param_str = f"epochs={params['n_epochs']}, reg_pu={params['reg_pu']}, reg_qi={params['reg_qi']}"
                else:
                    param_str = f"epochs={params['n_epochs']}, lr={params.get('lr_all', 'N/A')}, reg={params['reg_all']}"
                param_combos.append(param_str)
                rmse_means.append(cv_results['mean_test_rmse'][i])
                mae_means.append(cv_results['mean_test_mae'][i])

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'first_round', model_name, best_params, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            print(f"Error during optimization: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Optimization failed: {err}"))

    def _run_manual_grid_search(self, model_name, epochs, lr_values, reg_values, models):
        """Manual implementation of grid search when GridSearchCV fails"""
        try:
            print("Running manual grid search...")
            algo_class = models[model_name]
            best_rmse = float('inf')
            best_mae = float('inf')
            best_params = None

            # Split data
            trainset, testset = train_test_split(self.data, test_size=0.25)

            # Store results for visualization
            param_combos = []
            rmse_means = []
            mae_means = []

            # Try each parameter combination
            for n_epochs in epochs:
                for lr in lr_values:
                    for reg in reg_values:
                        try:
                            print(f"Testing params: epochs={n_epochs}, lr={lr}, reg={reg}")

                            # Create and train model
                            model = algo_class(n_epochs=n_epochs, lr_all=lr, reg_all=reg)
                            model.fit(trainset)

                            # Test model
                            predictions = model.test(testset)
                            rmse = accuracy.rmse(predictions)
                            mae = accuracy.mae(predictions)

                            print(f"RMSE: {rmse}, MAE: {mae}")

                            # Store results for this combination
                            param_str = f"epochs={n_epochs}, lr={lr}, reg={reg}"
                            param_combos.append(param_str)
                            rmse_means.append(rmse)
                            mae_means.append(mae)

                            # Update best parameters
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_mae = mae
                                best_params = {'n_epochs': n_epochs, 'lr_all': lr, 'reg_all': reg}
                        except Exception as e:
                            print(f"Error with params (epochs={n_epochs}, lr={lr}, reg={reg}): {e}")
                            # Add dummy results for failed combinations
                            param_str = f"epochs={n_epochs}, lr={lr}, reg={reg} (failed)"
                            param_combos.append(param_str)
                            rmse_means.append(None)
                            mae_means.append(None)

            if best_params is None:
                raise Exception("All parameter combinations failed")

            # Store results
            self.best_params['first_round'] = best_params
            self.current_model_name = model_name

            print(f"Manual grid search complete. Best params: {best_params}, RMSE: {best_rmse}, MAE: {best_mae}")

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'first_round', model_name, best_params, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            print(f"Error in manual grid search: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Manual optimization failed: {err}"))

    def _update_optimization_results(self, round_name, model_name, best_params, best_rmse, best_mae,
                                     param_combos, rmse_means, mae_means):
        """Update UI with optimization results"""
        # Update results text
        self.opt_results_text.config(state=tk.NORMAL)

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results_text = f"=== {round_name.replace('_', ' ').title()} ===\n"
        results_text += f"Time: {timestamp}\n"
        results_text += f"Model: {model_name}\n\n"
        results_text += f"Best Parameters:\n"

        for param, value in best_params.items():
            results_text += f"  {param}: {value}\n"

        results_text += f"\nBest RMSE: {best_rmse:.4f}\n"
        results_text += f"Best MAE: {best_mae:.4f}\n\n"

        self.opt_results_text.insert(tk.END, results_text)
        self.opt_results_text.see(tk.END)
        self.opt_results_text.config(state=tk.DISABLED)

        # Update visualization
        self._update_optimization_visualization(param_combos, rmse_means, mae_means)

        # Update status
        self.update_status(f"Optimization complete. Best RMSE: {best_rmse:.4f}")

    def _update_optimization_visualization(self, param_combos, rmse_means, mae_means):
        """Update visualization for optimization results"""
        # Clear previous chart
        for widget in self.opt_fig_frame.winfo_children():
            widget.destroy()

        # Create new figure with two subplots
        fig = Figure(figsize=(10, 6))

        # RMSE plot
        ax1 = fig.add_subplot(211)
        ax1.plot(range(len(param_combos)), rmse_means, 'o-', color=self.primary_color)
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE for Different Parameter Combinations')
        ax1.set_xticks([])  # Hide x ticks for this subplot

        # Find and mark best (minimum) RMSE
        best_idx = np.argmin(rmse_means)
        ax1.plot(best_idx, rmse_means[best_idx], 'o', color='red', markersize=10)
        ax1.annotate(f'Best: {rmse_means[best_idx]:.4f}',
                     xy=(best_idx, rmse_means[best_idx]),
                     xytext=(10, -20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

        # MAE plot
        ax2 = fig.add_subplot(212)
        ax2.plot(range(len(param_combos)), mae_means, 'o-', color=self.accent_color)
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE for Different Parameter Combinations')

        # X axis with parameter combinations (shortened for readability)
        short_params = [p[:20] + '...' if len(p) > 20 else p for p in param_combos]
        ax2.set_xticks(range(len(param_combos)))
        ax2.set_xticklabels(short_params, rotation=45, ha='right')

        # Find and mark best (minimum) MAE
        best_idx = np.argmin(mae_means)
        ax2.plot(best_idx, mae_means[best_idx], 'o', color='red', markersize=10)
        ax2.annotate(f'Best: {mae_means[best_idx]:.4f}',
                     xy=(best_idx, mae_means[best_idx]),
                     xytext=(10, -20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))

        fig.tight_layout()

        # Add chart to frame
        canvas = FigureCanvasTkAgg(fig, master=self.opt_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_second_optimization(self):
        """Run second round of hyperparameter optimization with factors"""
        if 'first_round' not in self.best_params:
            messagebox.showwarning("Warning", "Please run first optimization round before proceeding.")
            return

        # Get model and parameters
        model_name = self.current_model_name
        best_params = self.best_params['first_round']

        # Parse factors range
        try:
            factors = [int(x.strip()) for x in self.factors_var.get().split(',')]
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter comma-separated numbers.")
            return

        self.update_status(f"Running second optimization round for {model_name}...", start_progress=True)

        # Ensure fresh model imports
        from surprise import NMF, SVD, SVDpp

        # Initialize the parameter grid
        param_grid = {'n_factors': factors}

        # Handle models that require lr_all and reg_all (SVD and SVDpp)
        if model_name in ["SVD", "SVDpp"]:
            try:
                lr_values = [float(x.strip()) for x in self.lr_var.get().split(',')]
                reg_values = [float(x.strip()) for x in self.reg_var.get().split(',')]
                param_grid.update({
                    'lr_all': lr_values,
                    'reg_all': reg_values
                })
            except ValueError:
                messagebox.showerror("Error", "Invalid parameter values for learning rate and regularization.")
                return
        elif model_name == "NMF":
            # For NMF, we only need n_factors and n_epochs
            param_grid['n_epochs'] = [best_params['n_epochs']]  # Use the best n_epochs from first round

        # Create GridSearchCV with the parameter grid
        try:
            # Create a dictionary to map model names to their classes
            algo_class = {"SVD": SVD, "SVDpp": SVDpp, "NMF": NMF}[model_name]

            # Create GridSearchCV without unnecessary params
            gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=3)
            gs.fit(self.data)

            # Extract best params and scores
            best_params = gs.best_params['rmse']
            best_rmse = gs.best_score['rmse']
            best_mae = gs.best_score['mae']

            # Store results
            self.best_params['second_round'] = best_params

            # Prepare data for visualization
            param_combos = []
            rmse_means = []
            mae_means = []

            for i, params in enumerate(gs.cv_results['params']):
                param_str = f"factors={params['n_factors']}, epochs={best_params['n_epochs']}"
                if model_name in ["SVD", "SVDpp"]:
                    param_str += f", lr={params['lr_all']}, reg={params['reg_all']}"
                param_combos.append(param_str)
                rmse_means.append(gs.cv_results['mean_test_rmse'][i])
                mae_means.append(gs.cv_results['mean_test_mae'][i])

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'second_round', model_name, best_params, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            self.update_status(f"Error during second optimization: {str(e)}")

    def _run_second_optimization_thread(self, model_name, best_params, factors):
        """Background thread for second optimization round"""
        try:
            print(f"Starting second optimization for model: {model_name}")
            print(f"Best parameters from first round: {best_params}")
            print(f"Factors to test: {factors}")

            # Import necessary modules
            from surprise import SVD, SVDpp, NMF
            from surprise.model_selection import train_test_split
            from surprise import accuracy

            # Split data for evaluation
            trainset, testset = train_test_split(self.data, test_size=0.25, random_state=42)

            # Initialize variables to track best results
            best_rmse = float('inf')
            best_mae = float('inf')
            best_params_second = None

            # Store results for visualization
            param_combos = []
            rmse_means = []
            mae_means = []

            # Get parameters from first round based on model type
            n_epochs = best_params['n_epochs']

            # Handle model-specific parameters
            if model_name == "NMF":
                # NMF uses reg_pu and reg_qi instead of reg_all, and doesn't use lr_all
                reg_pu = best_params.get('reg_pu', 0.06)
                reg_qi = best_params.get('reg_qi', 0.06)
                lr_all = None  # NMF doesn't use lr_all
            elif model_name in ["SVD", "SVDpp"]:
                # SVD and SVDpp use both lr_all and reg_all
                lr_all = best_params.get('lr_all')
                reg_all = best_params.get('reg_all')
                reg_pu = None
                reg_qi = None
            else:
                # For other models that might not use these parameters
                lr_all = None
                reg_all = None
                reg_pu = None
                reg_qi = None

            # Try each factor value
            for n_factors in factors:
                try:
                    print(f"Testing n_factors={n_factors}")

                    # Create and train model based on model name
                    if model_name == "SVD":
                        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "SVDpp":
                        algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "NMF":
                        # NMF doesn't use lr_all parameter and uses reg_pu, reg_qi instead of reg_all
                        algo = NMF(n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu, reg_qi=reg_qi)
                    else:
                        print(f"Unknown model: {model_name}, using SVD as fallback")
                        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

                    # Train model
                    algo.fit(trainset)

                    # Test model
                    predictions = algo.test(testset)
                    rmse = accuracy.rmse(predictions)
                    mae = accuracy.mae(predictions)

                    print(f"n_factors={n_factors}: RMSE={rmse:.4f}, MAE={mae:.4f}")

                    # Store results for this combination
                    if model_name == "NMF":
                        param_str = f"factors={n_factors}, epochs={n_epochs}, reg_pu={reg_pu}, reg_qi={reg_qi}"
                    else:
                        param_str = f"factors={n_factors}, epochs={n_epochs}, lr={lr_all}, reg={reg_all}"

                    param_combos.append(param_str)
                    rmse_means.append(rmse)
                    mae_means.append(mae)

                    # Update best parameters if this is better
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_mae = mae
                        best_params_second = {
                            'n_factors': n_factors,
                            'n_epochs': n_epochs
                        }

                        # Add model-specific parameters
                        if model_name == "NMF":
                            best_params_second['reg_pu'] = reg_pu
                            best_params_second['reg_qi'] = reg_qi
                        elif model_name in ["SVD", "SVDpp"]:
                            best_params_second['lr_all'] = lr_all
                            best_params_second['reg_all'] = reg_all

                except Exception as e:
                    print(f"Error with n_factors={n_factors}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add dummy results for failed combinations
                    param_str = f"factors={n_factors} (failed)"
                    param_combos.append(param_str)
                    rmse_means.append(None)
                    mae_means.append(None)

            if best_params_second is None:
                raise Exception("All factor values failed")

            print(f"Best parameters found: {best_params_second}")
            print(f"Best RMSE: {best_rmse}, Best MAE: {best_mae}")

            # Store results
            self.best_params['second_round'] = best_params_second

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'second_round', model_name, best_params_second, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            print(f"Error in second optimization thread: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Second optimization failed: {err}"))

        try:
            print("Running manual grid search for factors...")

            # Ensure the necessary classes are imported
            from surprise import SVD, SVDpp, NMF
            from surprise.model_selection import train_test_split
            from surprise import accuracy

            # Split data
            trainset, testset = train_test_split(self.data, test_size=0.25)

            best_rmse = float('inf')
            best_mae = float('inf')
            best_params_second = None

            # Store results for visualization
            param_combos = []
            rmse_means = []
            mae_means = []

            # Get parameters from first round
            n_epochs = best_params['n_epochs']

            # Only get these parameters if applicable to the model
            lr_all = best_params.get('lr_all') if model_name != "NMF" else None
            reg_all = best_params.get('reg_all')

            # Try each factor value
            for n_factors in factors:
                try:
                    print(f"Testing n_factors={n_factors}")

                    # Create and train model based on model type
                    if model_name == "SVD":
                        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "SVDpp":
                        model = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "NMF":
                        # NMF doesn't use lr_all parameter
                        model = NMF(n_factors=n_factors, n_epochs=n_epochs)
                    else:
                        print(f"Unknown model: {model_name}, using SVD as fallback")
                        model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

                    model.fit(trainset)

                    # Test model
                    predictions = model.test(testset)
                    rmse = accuracy.rmse(predictions)
                    mae = accuracy.mae(predictions)

                    print(f"RMSE: {rmse}, MAE: {mae}")

                    # Store results for this combination
                    param_str = f"factors={n_factors}"
                    param_combos.append(param_str)
                    rmse_means.append(rmse)
                    mae_means.append(mae)

                    # Update best parameters
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_mae = mae
                        best_params_second = {
                            'n_factors': n_factors,
                            'n_epochs': n_epochs
                        }
                        if model_name != "NMF" and lr_all is not None:
                            best_params_second['lr_all'] = lr_all
                        if reg_all is not None:
                            best_params_second['reg_all'] = reg_all
                except Exception as e:
                    print(f"Error with n_factors={n_factors}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add dummy results for failed combinations
                    param_str = f"factors={n_factors} (failed)"
                    param_combos.append(param_str)
                    rmse_means.append(None)
                    mae_means.append(None)

            if best_params_second is None:
                raise Exception("All factor values failed")

            # Store results
            self.best_params['second_round'] = best_params_second

            print(f"Manual grid search complete. Best params: {best_params_second}, RMSE: {best_rmse}, MAE: {best_mae}")

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'second_round', model_name, best_params_second, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            print(f"Error in manual grid search: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Manual optimization failed: {err}"))

    def train_final_model(self):
        """Train final model with best parameters"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get selected model name
        model_name = self.model_var.get()
        self.current_model_name = model_name

        # Check if we have optimized parameters
        use_optimized = False
        params = None
        if 'second_round' in self.best_params and self.current_model_name in ["SVD", "SVDpp", "NMF"]:
            params = self.best_params['second_round']
            use_optimized = True
        elif 'first_round' in self.best_params:
            params = self.best_params['first_round']
            use_optimized = True

        self.update_status(f"Training {model_name} model...", start_progress=True)

        # Start training in a separate thread
        threading.Thread(target=self._train_final_model_thread,
                         args=(model_name, use_optimized, params if use_optimized else None)).start()

    def save_model(self):
        """Save the trained model to a file"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model to save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{self.current_model_name}_model_{timestamp}.pkl"

        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            initialfile=default_filename,
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Save model and related data
                model_data = {
                    'model': self.model,
                    'model_name': self.current_model_name,
                    'timestamp': timestamp,
                    'metrics': {
                        'rmse': self.model.rmse if hasattr(self.model, 'rmse') else None,
                        'mae': self.model.mae if hasattr(self.model, 'mae') else None
                    }
                }

                with open(file_path, 'wb') as f:
                    pickle.dump(model_data, f)

                # Store the path for quick access
                self.last_saved_path = file_path

                self.update_status(f"Model saved to {file_path}")
                messagebox.showinfo("Save Successful", f"Model saved to {file_path}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        """Load a trained model from a file"""
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.update_status(f"Loading model from {file_path}...", start_progress=True)

                # Load model in a separate thread to avoid UI freezing
                threading.Thread(target=self._load_model_thread, args=(file_path,)).start()

            except Exception as e:
                self.update_status("Ready")
                messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")

    def _load_model_thread(self, file_path):
        """Background thread for loading model"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            # Extract model and metadata
            self.model = model_data['model']
            self.current_model_name = model_data.get('model_name', 'Unknown')

            # Update UI in main thread
            self.root.after(0, lambda: self._update_model_ui(model_data))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Failed to load model: {err}"))

    def _update_model_ui(self, model_data):
        """Update UI after loading a model"""
        # Update model info
        timestamp = model_data.get('timestamp', 'Unknown')
        metrics = model_data.get('metrics', {})
        rmse = metrics.get('rmse', 'N/A')
        mae = metrics.get('mae', 'N/A')

        info_text = f"Model: {self.current_model_name} | Loaded from file"
        if rmse != 'N/A':
            info_text += f" | RMSE: {rmse:.4f}"
        if mae != 'N/A':
            info_text += f" | MAE: {mae:.4f}"

        self.model_info_label.config(text=info_text)

        # Update status
        self.update_status(f"Model {self.current_model_name} loaded successfully")

        # Enable recommendation features
        if hasattr(self, 'recommend_button'):
            self.recommend_button.config(state=tk.NORMAL)

    def _train_final_model_thread(self, model_name, use_optimized, params=None):
        """Background thread for training final model"""
        try:
            # Create model - FIXED VERSION
            if model_name == "SVD":
                if use_optimized:
                    algo = SVD(n_factors=params.get('n_factors', 100),
                               n_epochs=params.get('n_epochs', 20),
                               lr_all=params.get('lr_all', 0.005),
                               reg_all=params.get('reg_all', 0.02))
                else:
                    algo = SVD()
            elif model_name == "SVDpp":
                if use_optimized:
                    algo = SVDpp(n_factors=params.get('n_factors', 100),
                                 n_epochs=params.get('n_epochs', 20),
                                 lr_all=params.get('lr_all', 0.005),
                                 reg_all=params.get('reg_all', 0.02))
                else:
                    algo = SVDpp()
            elif model_name == "NMF":
                if use_optimized:
                    # NMF uses separate regularization parameters
                    algo = NMF(n_factors=params.get('n_factors', 15),
                               n_epochs=params.get('n_epochs', 50),
                               reg_pu=params.get('reg_pu', 0.06),
                               reg_qi=params.get('reg_qi', 0.06))
                else:
                    algo = NMF()
            elif model_name == "KNNBasic":
                algo = KNNBasic()
            elif model_name == "KNNWithMeans":
                algo = KNNWithMeans()
            elif model_name == "SlopeOne":
                algo = SlopeOne()
            elif model_name == "CoClustering":
                algo = CoClustering()
            else:
                algo = SVD()  # Default fallback

            # Split data and train
            trainset, testset = train_test_split(self.data, test_size=0.25)
            self.trainset = trainset
            self.testset = testset

            start_time = datetime.datetime.now()
            algo.fit(trainset)
            train_time = (datetime.datetime.now() - start_time).total_seconds()

            # Test on test set
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)

            # Store model and results
            self.model = algo

            # Store predictions for best/worst analysis
            test_predictions = []
            for uid, iid, true_r, est, _ in predictions:
                test_predictions.append((uid, iid, true_r, est, abs(true_r - est)))

            # Sort by error (ascending for best, descending for worst)
            best_predictions = sorted(test_predictions, key=lambda x: x[4])[:10]
            worst_predictions = sorted(test_predictions, key=lambda x: x[4], reverse=True)[:10]

            # Update UI in main thread
            self.root.after(0, lambda: self._update_final_model_ui(
                model_name, rmse, mae, train_time, use_optimized, best_predictions, worst_predictions))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Training failed: {err}"))

    def _update_final_model_ui(self, model_name, rmse, mae, train_time, use_optimized,
                               best_predictions, worst_predictions):
        """Update UI after final model training"""
        # Update model info
        info_text = f"Model: {model_name} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Train Time: {train_time:.2f}s"
        if use_optimized:
            info_text += " | Using optimized parameters"

        self.model_info_label.config(text=info_text)

        # Update best/worst prediction trees
        # Clear previous entries
        for tree in [self.best_tree, self.worst_tree]:
            for item in tree.get_children():
                tree.delete(item)

        # Insert best predictions
        for uid, iid, true_r, est, error in best_predictions:
            self.best_tree.insert('', 'end', values=(uid, iid, f"{true_r:.1f}", f"{est:.2f}", f"{error:.4f}"))

        # Insert worst predictions
        for uid, iid, true_r, est, error in worst_predictions:
            self.worst_tree.insert('', 'end', values=(uid, iid, f"{true_r:.1f}", f"{est:.2f}", f"{error:.4f}"))

        self.update_status(f"{model_name} model trained successfully. RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    def get_recommendations(self):
        """Get recommendations for selected user"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first.")
            return

        # Get selected user
        user_id = self.user_var.get()
        if not user_id:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        self.update_status(f"Generating recommendations for user {user_id}...")

        # Clear previous recommendations
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)

        try:
            # Get all items this user hasn't rated
            user_items = set(self.df[self.df['user_id'] == user_id]['item_id'])
            all_items = set(self.df['item_id'].unique())
            items_to_predict = list(all_items - user_items)

            # If too many items, sample a reasonable number
            if len(items_to_predict) > 1000:
                items_to_predict = random.sample(items_to_predict, 1000)

            # Generate predictions
            predictions = []
            for item_id in items_to_predict:
                est = self.model.predict(user_id, item_id).est
                predictions.append((item_id, est))

            # Sort by predicted rating (descending)
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Store for potential export
            self.recommendations = predictions[:20]

            # Display top recommendations
            for i, (item_id, rating) in enumerate(predictions[:20]):
                self.rec_tree.insert('', 'end', values=(item_id, f"{rating:.2f}"))

            self.update_status(f"Generated {len(predictions)} recommendations for user {user_id}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {str(e)}")
            self.update_status("Error generating recommendations")

    def export_recommendations(self):
        """Export current recommendations to a file"""
        if not hasattr(self, 'recommendations') or not self.recommendations:
            messagebox.showwarning("Warning", "No recommendations to export.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"recommendations_{timestamp}.csv"

        file_path = filedialog.asksaveasfilename(
            title="Export Recommendations",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Create a DataFrame from recommendations
                recommendations_df = pd.DataFrame(self.recommendations, columns=['item_id', 'predicted_rating'])
                recommendations_df.to_csv(file_path, index=False)

                self.update_status(f"Recommendations exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Recommendations exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export recommendations: {str(e)}")

    def find_similar_items(self):
        """Find similar items to the selected item"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first.")
            return

        # Get selected item
        item_id = self.item_var.get()
        if not item_id:
            messagebox.showwarning("Warning", "Please select an item.")
            return

        # This only works for certain model types that have item factors
        if self.current_model_name not in ["SVD", "SVDpp", "NMF"]:
            messagebox.showinfo("Information",
                                f"The {self.current_model_name} model doesn't support item similarity computation. Please use SVD, SVD++, or NMF.")
            return

        self.update_status(f"Finding items similar to {item_id}...")

        # Clear previous similar items
        if hasattr(self, 'similar_tree'):
            for item in self.similar_tree.get_children():
                self.similar_tree.delete(item)

        try:
            # Use model to find similar items
            # This is a simplified approach - in a real implementation,
            # we would use the item factors from the model

            # For matrix factorization models, we can use the item factors
            if hasattr(self.model, 'qi') and item_id in self.model.trainset.to_inner_iid:
                inner_id = self.model.trainset.to_inner_iid(item_id)
                item_factors = self.model.qi[inner_id]

                # Calculate similarity to all other items
                similarities = []
                for other_item in self.df['item_id'].unique():
                    if other_item != item_id and other_item in self.model.trainset.to_inner_iid:
                        other_inner_id = self.model.trainset.to_inner_iid(other_item)
                        other_factors = self.model.qi[other_inner_id]

                        # Cosine similarity
                        sim = np.dot(item_factors, other_factors) / (
                                np.linalg.norm(item_factors) * np.linalg.norm(other_factors))
                        similarities.append((other_item, sim))

                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Display top similar items
                for i, (other_item, sim) in enumerate(similarities[:10]):
                    self.similar_tree.insert('', 'end', values=(other_item, f"{sim:.4f}"))

                self.update_status(f"Found similar items to {item_id}")
            else:
                # Fallback to a simple collaborative approach
                # Get users who rated this item highly
                item_ratings = self.df[self.df['item_id'] == item_id]
                if len(item_ratings) == 0:
                    messagebox.showinfo("Information", f"No ratings found for item {item_id}")
                    return

                high_ratings = item_ratings[item_ratings['rating'] > item_ratings['rating'].mean()]
                if len(high_ratings) == 0:
                    high_ratings = item_ratings  # Use all ratings if none are above average

                users = high_ratings['user_id'].unique()

                # Find other items these users rated highly
                other_items = {}
                for user in users:
                    user_ratings = self.df[(self.df['user_id'] == user) & (self.df['item_id'] != item_id)]
                    for _, row in user_ratings.iterrows():
                        other_item = row['item_id']
                        rating = row['rating']
                        if other_item not in other_items:
                            other_items[other_item] = []
                        other_items[other_item].append(rating)

                # Calculate average rating for each item
                similarities = []
                for other_item, ratings in other_items.items():
                    avg_rating = sum(ratings) / len(ratings)
                    count = len(ratings)  # How many users rated both items
                    similarities.append((other_item, avg_rating, count))

                # Sort by count (more shared users) and then by average rating
                similarities.sort(key=lambda x: (x[2], x[1]), reverse=True)

                # Display top similar items
                for i, (other_item, avg_rating, count) in enumerate(similarities[:10]):
                    similarity_score = avg_rating / 5.0  # Normalize to 0-1 scale
                    self.similar_tree.insert('', 'end', values=(other_item, f"{similarity_score:.4f} ({count} users)"))

                self.update_status(f"Found similar items to {item_id} using collaborative approach")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find similar items: {str(e)}")
            self.update_status("Error finding similar items")

    def run_evaluation(self):
        """Run comprehensive model evaluation"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get parameters
        model_name = self.eval_model_var.get()
        cv_folds = self.cv_folds_var.get()
        test_size = self.test_size_var.get()
        seed = self.eval_seed_var.get()
        k = self.k_var.get()

        # Get selected metrics
        metrics = {key: var.get() for key, var in self.metric_vars.items()}

        if not any(metrics.values()):
            messagebox.showwarning("Warning", "Please select at least one metric to evaluate.")
            return

        self.update_status(f"Evaluating {model_name}...", start_progress=True)

        # Start evaluation in a separate thread
        threading.Thread(target=self._run_evaluation_thread,
                         args=(model_name, cv_folds, test_size, seed, k, metrics)).start()

    def _run_evaluation_thread(self, model_name, cv_folds, test_size, seed, k, metrics):
        """Background thread for model evaluation"""
        try:
            from surprise import SVD, SVDpp, SlopeOne, NMF, CoClustering, KNNBasic, KNNWithMeans, KNNWithZScore

            # Create algorithm instance - FIXED VERSION
            if model_name == "SVD":
                algo = SVD()  # Create a new SVD instance
            elif model_name == "SVDpp":
                algo = SVDpp()
            elif model_name == "NMF":
                algo = NMF()
            elif model_name == "SlopeOne":
                algo = SlopeOne()
            elif model_name == "CoClustering":
                algo = CoClustering()
            elif model_name == "KNNBasic":
                algo = KNNBasic()
            elif model_name == "KNNWithMeans":
                algo = KNNWithMeans()
            elif model_name == "KNNWithZScore":
                algo = KNNWithZScore()
            else:
                raise ValueError(f"Unknown algorithm: {model_name}")

            # Debug print
            print(f"Created evaluation algorithm {model_name}: {type(algo)}")

            # Split data for holdout evaluation
            trainset, testset = train_test_split(self.data, test_size=test_size, random_state=seed)

            # Train model
            algo.fit(trainset)

            # Test on test set
            predictions = algo.test(testset)

            # Calculate standard metrics
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)

            # Store evaluation results
            results = {
                'rmse': rmse,
                'mae': mae,
                'model': model_name,
                'test_size': test_size,
                'seed': seed
            }

            # Add to metrics dict
            if model_name not in self.evaluation_metrics:
                self.evaluation_metrics[model_name] = {}
            self.evaluation_metrics[model_name].update(results)

            # Additional metrics if selected
            if metrics.get('precision') or metrics.get('recall') or metrics.get('f1'):
                precision, recall, f1 = self._calculate_prec_rec_metrics(predictions, k)
                results.update({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'k': k
                })
                self.evaluation_metrics[model_name].update(results)

            if metrics.get('coverage'):
                coverage = self._calculate_coverage(algo, trainset, testset)
                results['coverage'] = coverage
                self.evaluation_metrics[model_name].update(results)

            # Cross-validation if requested
            if cv_folds > 0:
                cv_results = cross_validate(algo, self.data, measures=['RMSE', 'MAE'], cv=cv_folds, verbose=False)
                results.update({
                    'cv_rmse_mean': np.mean(cv_results['test_rmse']),
                    'cv_rmse_std': np.std(cv_results['test_rmse']),
                    'cv_mae_mean': np.mean(cv_results['test_mae']),
                    'cv_mae_std': np.std(cv_results['test_mae']),
                    'cv_folds': cv_folds
                })
                self.evaluation_metrics[model_name].update(results)

            # Prepare data for ROC and PR curves
            if metrics.get('precision') or metrics.get('recall'):
                # Get positive/negative classes for ROC
                y_true, y_scores = self._prepare_roc_data(predictions)
                self.precision_recall_data[model_name] = {
                    'y_true': y_true,
                    'y_scores': y_scores
                }

            # Update UI in main thread
            self.root.after(0, lambda: self._update_evaluation_results(model_name, results))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Evaluation failed: {err}"))

    def _calculate_prec_rec_metrics(self, predictions, k):
        """Calculate precision, recall, and F1 at k"""
        # Group predictions by user
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = []
        recalls = []
        f1_scores = []

        for uid, user_ratings in user_est_true.items():
            # Sort by estimated rating
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Get top k recommendations
            n_rel = sum((true_r >= 4.0) for (_, true_r) in user_ratings)
            n_rec_k = min(k, len(user_ratings))
            n_rel_and_rec_k = sum((true_r >= 4.0) for (_, true_r) in user_ratings[:n_rec_k])

            # Precision@k: proportion of recommended items that are relevant
            precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@k: proportion of relevant items that are recommended
            recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

            # F1@k
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Average metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)

        return avg_precision, avg_recall, avg_f1

    def _calculate_coverage(self, algo, trainset, testset):
        """Calculate catalog coverage of the algorithm"""
        all_items = set(trainset.all_items())
        rec_items = set()

        # For each user, get top-k recommendations
        k = 10  # Number of recommendations per user
        users = list(trainset.all_users())

        # Sample users if too many
        if len(users) > 100:
            users = random.sample(users, 100)

        for user in users:
            # Get the raw user ID
            raw_uid = trainset.to_raw_uid(user)

            # Get all items this user hasn't rated in the training set
            user_items = set(j for (j, _) in trainset.ur[user])
            candidate_items = list(all_items - user_items)

            # Convert to raw item IDs
            raw_candidate_items = [trainset.to_raw_iid(item) for item in candidate_items]

            # If too many items, sample them
            if len(raw_candidate_items) > 1000:
                raw_candidate_items = random.sample(raw_candidate_items, 1000)

            # Predict ratings and get top-k
            item_ratings = []
            for item in raw_candidate_items:
                try:
                    pred = algo.predict(raw_uid, item).est
                    item_ratings.append((item, pred))
                except:
                    continue

            # Sort by predicted rating and get top-k
            item_ratings.sort(key=lambda x: x[1], reverse=True)
            for item, _ in item_ratings[:k]:
                rec_items.add(item)

        # Calculate coverage
        coverage = len(rec_items) / len(all_items) if all_items else 0
        return coverage

    def _prepare_roc_data(self, predictions):
        """Prepare data for ROC and PR curves"""
        # Define positive class as ratings >= 4
        y_true = []
        y_scores = []

        for _, _, true_r, est, _ in predictions:
            # Convert true ratings to binary (positive if >= 4)
            y_true.append(1 if true_r >= 4.0 else 0)
            y_scores.append(est)

        return np.array(y_true), np.array(y_scores)

    def _update_evaluation_results(self, model_name, results):
        """Update UI with evaluation results"""
        # Update text area
        self.eval_results_text.config(state=tk.NORMAL)
        self.eval_results_text.delete(1.0, tk.END)

        result_text = f"Model: {model_name}\n"
        result_text += f"Test Size: {results['test_size']}\n"
        result_text += f"Random Seed: {results['seed']}\n\n"

        result_text += "Basic Metrics:\n"
        result_text += f"RMSE: {results['rmse']:.4f}\n"
        result_text += f"MAE: {results['mae']:.4f}\n\n"

        if 'precision' in results:
            result_text += f"Ranking Metrics (k={results.get('k', 10)}):\n"
            result_text += f"Precision@k: {results['precision']:.4f}\n"
            result_text += f"Recall@k: {results['recall']:.4f}\n"
            result_text += f"F1@k: {results['f1']:.4f}\n\n"

        if 'coverage' in results:
            result_text += f"Coverage: {results['coverage']:.4f}\n\n"

        if 'cv_rmse_mean' in results:
            result_text += f"Cross-Validation ({results['cv_folds']} folds):\n"
            result_text += f"CV RMSE: {results['cv_rmse_mean']:.4f} ± {results['cv_rmse_std']:.4f}\n"
            result_text += f"CV MAE: {results['cv_mae_mean']:.4f} ± {results['cv_mae_std']:.4f}\n"

        self.eval_results_text.insert(tk.END, result_text)
        self.eval_results_text.config(state=tk.DISABLED)

        # Update visualizations
        self._update_evaluation_visualizations(model_name)

        self.update_status(f"Evaluation of {model_name} complete. RMSE: {results['rmse']:.4f}")

    def _update_evaluation_visualizations(self, model_name):
        """Update evaluation visualizations"""
        # Check if we have data for ROC/PR curves
        if model_name in self.precision_recall_data:
            data = self.precision_recall_data[model_name]
            y_true = data['y_true']
            y_scores = data['y_scores']

            # Update ROC curve
            self._plot_roc_curve(y_true, y_scores)

            # Update Precision-Recall curve
            self._plot_pr_curve(y_true, y_scores)

        # Update Cross-Validation results if available
        if model_name in self.evaluation_metrics:
            metrics = self.evaluation_metrics[model_name]
            if 'cv_rmse_mean' in metrics:
                self._plot_cv_results(model_name, metrics)

    def _plot_roc_curve(self, y_true, y_scores):
        """Plot ROC curve"""
        # Clear previous plot
        for widget in self.roc_frame.winfo_children():
            widget.destroy()

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Create plot
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Plot ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1)

        # Add labels and legend
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc='lower right')

        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=self.roc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_pr_curve(self, y_true, y_scores):
        """Plot Precision-Recall curve"""
        # Clear previous plot
        for widget in self.pr_frame.winfo_children():
            widget.destroy()

        # Compute Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Create plot
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        # Plot PR curve
        ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')

        # Add labels and legend
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')

        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=self.pr_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_cv_results(self, model_name, metrics):
        """Plot Cross-Validation results"""
        # Clear previous plot
        for widget in self.cv_frame.winfo_children():
            widget.destroy()

        # Create bar plot for RMSE and MAE
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        x = ['RMSE', 'MAE']
        y = [metrics['cv_rmse_mean'], metrics['cv_mae_mean']]
        yerr = [metrics['cv_rmse_std'], metrics['cv_mae_std']]

        # Create bars
        bars = ax.bar(x, y, yerr=yerr, alpha=0.7, capsize=10, color=[self.primary_color, self.accent_color])

        # Add labels
        ax.set_ylabel('Error')
        ax.set_title(f'Cross-Validation Results ({metrics["cv_folds"]} folds)')

        # Add value labels on bars
        for bar, val, err in zip(bars, y, yerr):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{val:.4f} ± {err:.4f}', ha='center', va='bottom', fontsize=9)

        # Add to frame
        canvas = FigureCanvasTkAgg(fig, master=self.cv_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_visualization(self):
        """Generate selected visualization"""
        if self.data is None or self.df is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        chart_type = self.chart_type_var.get()
        sample_size = self.sample_size_var.get()
        color_scheme = self.color_scheme_var.get()
        fig_width = self.fig_width_var.get()
        fig_height = self.fig_height_var.get()

        self.update_status(f"Generating {chart_type} visualization...", start_progress=True)

        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Create function-specific visualizations based on chart type
        try:
            # Create new figure
            fig = Figure(figsize=(fig_width, fig_height))

            if chart_type == "distribution":
                self._create_distribution_chart(fig, color_scheme)
            elif chart_type == "user_activity":
                self._create_user_activity_chart(fig, sample_size, color_scheme)
            elif chart_type == "item_popularity":
                self._create_item_popularity_chart(fig, sample_size, color_scheme)
            elif chart_type == "heatmap":
                self._create_rating_heatmap(fig, sample_size, color_scheme)
            elif chart_type == "user_similarity":
                self._create_user_similarity_chart(fig, sample_size, color_scheme)
            elif chart_type == "item_similarity":
                self._create_item_similarity_chart(fig, sample_size, color_scheme)

            # Add chart to frame
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Store current figure for export
            self.current_figure = fig

            self.update_status(f"{chart_type.replace('_', ' ').title()} visualization generated")

        except Exception as e:
            self.update_status("Ready")
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")

    def _create_distribution_chart(self, fig, color_scheme):
        """Create rating distribution chart"""
        ax = fig.add_subplot(111)

        # Get rating distribution
        rating_counts = self.df['rating'].value_counts().sort_index()

        # Create bar chart
        bars = sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax, palette=color_scheme)

        # Add percentage labels
        total = rating_counts.sum()
        for bar in bars.patches:
            height = bar.get_height()
            percentage = (height / total) * 100
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{percentage:.1f}%', ha='center', va='bottom')

        # Add labels and title
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Rating Distribution')

        fig.tight_layout()

    def _create_user_activity_chart(self, fig, sample_size, color_scheme):
        """Create user activity chart"""
        # Get ratings per user
        user_ratings = self.df.groupby('user_id').size().reset_index(name='rating_count')

        # Sample if too many users
        if len(user_ratings) > sample_size:
            user_ratings = user_ratings.sample(sample_size, random_state=42)

        # Sort by rating count
        user_ratings = user_ratings.sort_values('rating_count', ascending=False)

        # Create two subplots
        ax1 = fig.add_subplot(211)  # Top subplot for bar chart
        ax2 = fig.add_subplot(212)  # Bottom subplot for histogram

        # Bar chart of top users
        top_n = min(20, len(user_ratings))
        top_users = user_ratings.head(top_n)
        sns.barplot(x='user_id', y='rating_count', data=top_users, ax=ax1, palette=color_scheme)
        ax1.set_title(f'Top {top_n} Most Active Users')
        ax1.set_xlabel('User ID')
        ax1.set_ylabel('Number of Ratings')
        ax1.tick_params(axis='x', rotation=45)

        # Histogram of ratings per user
        sns.histplot(user_ratings['rating_count'], ax=ax2, bins=30, kde=True, color=self.primary_color)
        ax2.set_title('Distribution of Ratings per User')
        ax2.set_xlabel('Number of Ratings')
        ax2.set_ylabel('Number of Users')

        fig.tight_layout()

    def _create_item_popularity_chart(self, fig, sample_size, color_scheme):
        """Create item popularity chart"""
        # Get ratings per item
        item_ratings = self.df.groupby('item_id').size().reset_index(name='rating_count')
        item_avg_ratings = self.df.groupby('item_id')['rating'].mean().reset_index(name='avg_rating')

        # Merge count and average
        item_data = pd.merge(item_ratings, item_avg_ratings, on='item_id')

        # Sample if too many items
        if len(item_data) > sample_size:
            item_data = item_data.sample(sample_size, random_state=42)

        # Create two subplots
        ax1 = fig.add_subplot(211)  # Top subplot for bar chart
        ax2 = fig.add_subplot(212)  # Bottom subplot for scatter plot

        # Bar chart of top items by popularity
        top_n = min(20, len(item_data))
        top_items = item_data.sort_values('rating_count', ascending=False).head(top_n)
        sns.barplot(x='item_id', y='rating_count', data=top_items, ax=ax1, palette=color_scheme)
        ax1.set_title(f'Top {top_n} Most Popular Items')
        ax1.set_xlabel('Item ID')
        ax1.set_ylabel('Number of Ratings')
        ax1.tick_params(axis='x', rotation=45)

        # Scatter plot: popularity vs average rating
        sns.scatterplot(x='rating_count', y='avg_rating', data=item_data, ax=ax2, alpha=0.6,
                        hue='avg_rating', palette=color_scheme, legend=False)
        ax2.set_title('Item Popularity vs. Average Rating')
        ax2.set_xlabel('Number of Ratings (Popularity)')
        ax2.set_ylabel('Average Rating')

        fig.tight_layout()

    def _create_rating_heatmap(self, fig, sample_size, color_scheme):
        """Create rating heatmap"""
        # Sample users and items if dataset is too large
        users = self.df['user_id'].unique()
        items = self.df['item_id'].unique()

        # Take sample
        max_users = min(50, len(users))
        max_items = min(50, len(items))

        if len(users) > max_users:
            sampled_users = np.random.choice(users, max_users, replace=False)
        else:
            sampled_users = users

        if len(items) > max_items:
            sampled_items = np.random.choice(items, max_items, replace=False)
        else:
            sampled_items = items

        # Filter dataset
        sample_df = self.df[self.df['user_id'].isin(sampled_users) & self.df['item_id'].isin(sampled_items)]

        # Create pivot table for heatmap
        pivot_table = sample_df.pivot_table(values='rating', index='user_id', columns='item_id', fill_value=0)

        # Plot heatmap
        ax = fig.add_subplot(111)
        sns.heatmap(pivot_table, cmap=color_scheme, ax=ax)
        ax.set_title('User-Item Rating Heatmap')
        ax.set_xlabel('Item ID')
        ax.set_ylabel('User ID')

        fig.tight_layout()

    def _create_user_similarity_chart(self, fig, sample_size, color_scheme):
        """Create user similarity visualization"""
        if not hasattr(self, 'model') or self.model is None:
            # Add message to figure
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Train a matrix factorization model (SVD, SVD++, NMF) first\nto view user similarity chart",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        # Check if model has user factors
        if not hasattr(self.model, 'pu'):
            # Add message to figure
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f"The {self.current_model_name} model doesn't support user similarity visualization.\nUse SVD, SVD++, or NMF.",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        # Get user factors
        users = list(self.model.trainset.all_users())

        # Sample users if too many
        max_users = min(50, len(users))
        if len(users) > max_users:
            users = np.random.choice(users, max_users, replace=False)

        # Get factors for sampled users
        user_factors = np.array([self.model.pu[user] for user in users])

        # Convert inner user IDs to raw IDs for display
        user_ids = [self.model.trainset.to_raw_uid(user) for user in users]

        # Reduce dimensionality for visualization if factors > 2
        if user_factors.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            user_factors_2d = pca.fit_transform(user_factors)
        else:
            user_factors_2d = user_factors

        # Create scatter plot
        ax = fig.add_subplot(111)
        scatter = ax.scatter(user_factors_2d[:, 0], user_factors_2d[:, 1], c=range(len(users)),
                             cmap=color_scheme, alpha=0.7)

        # Add labels for some points
        max_labels = min(10, len(users))
        for i in np.random.choice(range(len(users)), max_labels, replace=False):
            ax.annotate(str(user_ids[i]), (user_factors_2d[i, 0], user_factors_2d[i, 1]))

        ax.set_title('User Similarity (2D Projection of User Factors)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        # Add colorbar
        fig.colorbar(scatter, ax=ax, label='User Index')

        fig.tight_layout()

    def _create_item_similarity_chart(self, fig, sample_size, color_scheme):
        """Create item similarity visualization"""
        if not hasattr(self, 'model') or self.model is None:
            # Add message to figure
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Train a matrix factorization model (SVD, SVD++, NMF) first\nto view item similarity chart",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        # Check if model has item factors
        if not hasattr(self.model, 'qi'):
            # Add message to figure
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    f"The {self.current_model_name} model doesn't support item similarity visualization.\nUse SVD, SVD++, or NMF.",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        # Get item factors
        items = list(self.model.trainset.all_items())

        # Sample items if too many
        max_items = min(50, len(items))
        if len(items) > max_items:
            items = np.random.choice(items, max_items, replace=False)

        # Get factors for sampled items
        item_factors = np.array([self.model.qi[item] for item in items])

        # Convert inner item IDs to raw IDs for display
        item_ids = [self.model.trainset.to_raw_iid(item) for item in items]

        # Reduce dimensionality for visualization if factors > 2
        if item_factors.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            item_factors_2d = pca.fit_transform(item_factors)
        else:
            item_factors_2d = item_factors

        # Create scatter plot
        ax = fig.add_subplot(111)
        scatter = ax.scatter(item_factors_2d[:, 0], item_factors_2d[:, 1], c=range(len(items)),
                             cmap=color_scheme, alpha=0.7)

        # Add labels for some points
        max_labels = min(10, len(items))
        for i in np.random.choice(range(len(items)), max_labels, replace=False):
            ax.annotate(str(item_ids[i]), (item_factors_2d[i, 0], item_factors_2d[i, 1]))

        ax.set_title('Item Similarity (2D Projection of Item Factors)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        # Add colorbar
        fig.colorbar(scatter, ax=ax, label='Item Index')

        fig.tight_layout()

    def export_data(self):
        """Export current data to a file"""
        if self.df is None:
            messagebox.showwarning("Warning", "No data to export.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"recommender_data_{timestamp}.csv"

        file_path = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Determine file type
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == '.csv':
                    self.df.to_csv(file_path, index=False)
                elif file_ext in ['.xlsx', '.xls']:
                    self.df.to_excel(file_path, index=False)
                else:
                    # Default to CSV
                    self.df.to_csv(file_path, index=False)

                self.update_status(f"Data exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Data exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def export_recommendations(self):
        """Export current recommendations to a file"""
        if not hasattr(self, 'recommendations') or not self.recommendations:
            messagebox.showwarning("Warning", "No recommendations to export. Generate recommendations first.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"recommendations_{timestamp}.csv"

        file_path = filedialog.asksaveasfilename(
            title="Export Recommendations",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Create a DataFrame from recommendations
                recommendations_df = pd.DataFrame(self.recommendations, columns=['item_id', 'predicted_rating'])

                # Determine file type
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == '.csv':
                    recommendations_df.to_csv(file_path, index=False)
                elif file_ext in ['.xlsx', '.xls']:
                    recommendations_df.to_excel(file_path, index=False)
                else:
                    # Default to CSV
                    recommendations_df.to_csv(file_path, index=False)

                self.update_status(f"Recommendations exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Recommendations exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export recommendations: {str(e)}")

    def export_chart(self):
        """Export current chart to a file"""
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("Warning", "No chart to export.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_type = self.chart_type_var.get()
        default_filename = f"chart_{chart_type}_{timestamp}.png"

        file_path = filedialog.asksaveasfilename(
            title="Export Chart",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.update_status(f"Chart exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Chart exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export chart: {str(e)}")

    def generate_synthetic_ecommerce_data(self, num_users=500, num_items=100, sparsity=0.9,
                                          distribution='skewed', seed=42):
        """Generate synthetic e-commerce dataset"""
        np.random.seed(seed)

        # Calculate how many ratings we'll generate
        num_ratings = int(num_users * num_items * (1 - sparsity))

        # Generate random users and items
        user_ids = np.random.choice(num_users, num_ratings)
        item_ids = np.random.choice(num_items, num_ratings)

        # Generate ratings based on selected distribution
        if distribution == 'skewed':
            # Skewed distribution (mostly high ratings, as in the paper)
            # From the pie chart in the paper: 54.6% are 5-star ratings, 24.6% are 4-star, etc.
            probabilities = [0.035, 0.059, 0.114, 0.246, 0.546]
            ratings = np.random.choice([1, 2, 3, 4, 5], num_ratings, p=probabilities)
        elif distribution == 'normal':
            # Normal distribution around 3.5 with std dev 1.0, clipped to 1-5 range
            ratings = np.clip(np.random.normal(3.5, 1.0, num_ratings), 1, 5).round().astype(int)
        else:  # uniform
            # Uniform distribution
            ratings = np.random.randint(1, 6, num_ratings)

        # Create DataFrame
        df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in user_ids],
            'item_id': [f'item_{i}' for i in item_ids],
            'rating': ratings
        })

        # Remove duplicates (same user rating same item multiple times)
        df = df.drop_duplicates(['user_id', 'item_id'])

        # Generate item features for content-based filtering
        # This could be categories, tags, etc.
        self.item_features = {}
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Beauty',
                      'Sports', 'Toys', 'Automotive', 'Grocery', 'Office']

        for item in range(num_items):
            item_id = f'item_{item}'
            # Assign random category and features
            category = np.random.choice(categories)
            price = round(np.random.uniform(5, 500), 2)
            popularity = np.random.uniform(1, 10)

            self.item_features[item_id] = {
                'category': category,
                'price': price,
                'popularity': popularity
            }

        return df

    def generate_data(self):
        """Generate synthetic data based on user parameters"""
        self.update_status("Generatgenerate_synthetic_ecommerce_dataing synthetic data...", start_progress=True)

        # Get parameters from UI
        num_users = self.num_users_var.get()
        num_items = self.num_items_var.get()
        sparsity = self.sparsity_var.get()
        distribution = self.rating_dist_var.get()
        seed = self.seed_var.get()

        try:
            # Run data generation in a separate thread
            thread = threading.Thread(target=self._generate_data_thread,
                                      args=(num_users, num_items, sparsity, distribution, seed))
            thread.start()
            self.active_threads.append(thread)
        except Exception as e:
            self.update_status(f"Error generating data: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def _generate_data_thread(self, num_users, num_items, sparsity, distribution, seed):
        """Background thread for data generation"""
        try:
            # Generate data
            self.df = self.generate_synthetic_ecommerce_data(
                num_users, num_items, sparsity, distribution, seed)

            # Convert to Surprise format
            reader = Reader(rating_scale=(1, 5))
            self.data = Dataset.load_from_df(self.df[['user_id', 'item_id', 'rating']], reader)

            # Update UI in main thread
            self.root.after(0, self._update_data_ui)
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Failed to generate data: {err}"))

    def _update_data_ui(self):
        """Update UI with generated data information"""
        # Update statistics text
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        # Calculate statistics
        total_ratings = len(self.df)
        unique_users = self.df['user_id'].nunique()
        unique_items = self.df['item_id'].nunique()
        rating_dist = self.df['rating'].value_counts(normalize=True).sort_index() * 100

        # Calculate density
        max_possible_ratings = unique_users * unique_items
        density = (total_ratings / max_possible_ratings) * 100

        # Additional statistics
        avg_rating = self.df['rating'].mean()
        median_rating = self.df['rating'].median()
        ratings_per_user = self.df.groupby('user_id').size()
        ratings_per_item = self.df.groupby('item_id').size()
        avg_ratings_per_user = ratings_per_user.mean()
        avg_ratings_per_item = ratings_per_item.mean()

        # Format statistics text
        stats = f"Total ratings: {total_ratings:,}\n"
        stats += f"Unique users: {unique_users:,}\n"
        stats += f"Unique items: {unique_items:,}\n"
        stats += f"Matrix density: {density:.2f}%\n\n"
        stats += f"Average rating: {avg_rating:.2f}\n"
        stats += f"Median rating: {median_rating:.1f}\n\n"
        stats += f"Avg ratings per user: {avg_ratings_per_user:.2f}\n"
        stats += f"Avg ratings per item: {avg_ratings_per_item:.2f}\n\n"
        stats += "Rating Distribution:\n"
        for rating, percentage in rating_dist.items():
            stats += f"Rating {rating}: {percentage:.2f}%\n"

        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)

        # Update visualization
        self._update_data_visualization()  # Make sure this line is present

        # Update dropdown values for user and item selection
        if hasattr(self, 'user_combo'):
            all_users = self.df['user_id'].unique().tolist()
            all_users.sort()
            self.user_combo['values'] = all_users[:100]  # Limit to first 100 for performance

        if hasattr(self, 'item_combo'):
            all_items = self.df['item_id'].unique().tolist()
            all_items.sort()
            self.item_combo['values'] = all_items[:100]  # Limit to first 100 for performance

        self.update_status(f"Data loaded: {total_ratings} ratings from {unique_users} users on {unique_items} items")

        # Update visualization
        def _update_data_visualization(self):
            """Update data visualization chart"""
            # Clear previous chart
            for widget in self.data_fig_frame.winfo_children():
                widget.destroy()

            # Create new figure
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)

            # Generate rating distribution chart
            rating_counts = self.df['rating'].value_counts().sort_index()
            bars = ax.bar(rating_counts.index, rating_counts.values, color=self.primary_color)

            # Add percentage labels on top of bars
            total = rating_counts.sum()
            for bar in bars:
                height = bar.get_height()
                percentage = (height / total) * 100
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{percentage:.1f}%', ha='center', va='bottom')

            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_title('Rating Distribution')
            ax.set_xticks(rating_counts.index)

            # Add chart to frame
            canvas = FigureCanvasTkAgg(fig, master=self.data_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Store current figure
            self.current_figure = fig

        def compare_algorithms(self):
            """Compare different recommendation algorithms"""
            if self.data is None:
                messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
                return

            # Get selected algorithms
            selected_algorithms = []
            for class_name, var in self.alg_vars.items():
                if var.get():
                    selected_algorithms.append(class_name)

            if not selected_algorithms:
                messagebox.showwarning("Warning", "Please select at least one algorithm to compare.")
                return

            self.update_status(f"Comparing {len(selected_algorithms)} algorithms...", start_progress=True)

            # Start comparison in a separate thread
            threading.Thread(target=self._compare_algorithms_thread,
                             args=(selected_algorithms,)).start()

        def _compare_algorithms_thread(self, selected_algorithms):
            """Background thread for algorithm comparison"""
            try:
                # Split data into train and test sets
                trainset, testset = train_test_split(self.data, test_size=0.25, random_state=42)

                # Create algorithm instances
                algorithms = {}
                for algo_name in selected_algorithms:
                    if algo_name == "SVD":
                        algorithms[algo_name] = SVD()
                    elif algo_name == "SVDpp":
                        algorithms[algo_name] = SVDpp()
                    elif algo_name == "NMF":
                        algorithms[algo_name] = NMF()
                    elif algo_name == "SlopeOne":
                        algorithms[algo_name] = SlopeOne()
                    elif algo_name == "CoClustering":
                        algorithms[algo_name] = CoClustering()
                    elif algo_name == "KNNBasic":
                        algorithms[algo_name] = KNNBasic()
                    elif algo_name == "KNNWithMeans":
                        algorithms[algo_name] = KNNWithMeans()
                    elif algo_name == "KNNWithZScore":
                        algorithms[algo_name] = KNNWithZScore()

                # Train and test each algorithm
                results = {}
                for name, algo in algorithms.items():
                    # Train
                    start_time = datetime.datetime.now()
                    algo.fit(trainset)
                    train_time = (datetime.datetime.now() - start_time).total_seconds()

                    # Test
                    start_time = datetime.datetime.now()
                    predictions = algo.test(testset)
                    test_time = (datetime.datetime.now() - start_time).total_seconds()

                    # Calculate metrics
                    rmse = accuracy.rmse(predictions)
                    mae = accuracy.mae(predictions)

                    # Store results
                    results[name] = {
                        'algorithm': algo,
                        'rmse': rmse,
                        'mae': mae,
                        'train_time': train_time,
                        'test_time': test_time,
                        'predictions': predictions
                    }

                    # Store model for later use
                    self.models[name] = algo

                # Store results
                self.results = results

                # Update UI in main thread
                self.root.after(0, lambda: self._update_comparison_results(results))

            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
                self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Comparison failed: {err}"))

        def _update_comparison_results(self, results):
            """Update UI with algorithm comparison results"""
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Sort algorithms by RMSE
            sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])

            # Insert results into tree
            for name, result in sorted_results:
                values = (
                    f"{result['rmse']:.4f}",
                    f"{result['mae']:.4f}",
                    f"{result['train_time']:.2f}",
                    f"{result['test_time']:.2f}"
                )
                self.results_tree.insert('', 'end', text=name, values=values, tags=(name,))

            # Highlight best algorithm
            best_algo = sorted_results[0][0]
            self.results_tree.tag_configure(best_algo, background=self.accent_color)

            # Update visualization
            self._update_comparison_visualization(results)

            # Update status
            self.update_status(
                f"Comparison complete. Best algorithm: {best_algo} (RMSE: {sorted_results[0][1]['rmse']:.4f})")

        def _update_comparison_visualization(self, results):
            """Update visualization charts for algorithm comparison"""
            # Clear previous charts
            for widget in self.rmse_chart_frame.winfo_children():
                widget.destroy()

            for widget in self.time_chart_frame.winfo_children():
                widget.destroy()

            for widget in self.metrics_chart_frame.winfo_children():
                widget.destroy()

            # Get data for charts
            names = list(results.keys())
            rmse_values = [results[name]['rmse'] for name in names]
            mae_values = [results[name]['mae'] for name in names]
            train_times = [results[name]['train_time'] for name in names]
            test_times = [results[name]['test_time'] for name in names]

            # RMSE Chart
            rmse_fig = Figure(figsize=(8, 5))
            rmse_ax = rmse_fig.add_subplot(111)

            bars = rmse_ax.bar(names, rmse_values, color=self.primary_color)
            rmse_ax.set_ylabel('RMSE (lower is better)')
            rmse_ax.set_title('RMSE Comparison')
            rmse_ax.set_xticklabels(names, rotation=45, ha='right')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                rmse_ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            rmse_canvas = FigureCanvasTkAgg(rmse_fig, master=self.rmse_chart_frame)
            rmse_canvas.draw()
            rmse_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Time Comparison Chart
            time_fig = Figure(figsize=(8, 5))
            time_ax = time_fig.add_subplot(111)

            x = np.arange(len(names))
            width = 0.35

            train_bars = time_ax.bar(x - width / 2, train_times, width, label='Training Time')
            test_bars = time_ax.bar(x + width / 2, test_times, width, label='Testing Time')

            time_ax.set_ylabel('Time (seconds)')
            time_ax.set_title('Performance Time Comparison')
            time_ax.set_xticks(x)
            time_ax.set_xticklabels(names, rotation=45, ha='right')
            time_ax.legend()

            time_canvas = FigureCanvasTkAgg(time_fig, master=self.time_chart_frame)
            time_canvas.draw()
            time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Metrics Comparison
            metrics_fig = Figure(figsize=(8, 5))
            metrics_ax = metrics_fig.add_subplot(111)

            x = np.arange(len(names))
            width = 0.35

            rmse_bars = metrics_ax.bar(x - width / 2, rmse_values, width, label='RMSE')
            mae_bars = metrics_ax.bar(x + width / 2, mae_values, width, label='MAE')

            metrics_ax.set_ylabel('Error Value')
            metrics_ax.set_title('Error Metrics Comparison')
            metrics_ax.set_xticks(x)
            metrics_ax.set_xticklabels(names, rotation=45, ha='right')
            metrics_ax.legend()

            metrics_canvas = FigureCanvasTkAgg(metrics_fig, master=self.metrics_chart_frame)
            metrics_canvas.draw()
            metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def run_first_optimization(self):
            """Run first round of hyperparameter optimization"""
            if self.data is None:
                messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
                return

            # Get selected model and parameters
            model_name = self.opt_model_var.get()

            # Parse parameter ranges
            try:
                epochs = [int(x.strip()) for x in self.epochs_var.get().split(',')]
                lr_values = [float(x.strip()) for x in self.lr_var.get().split(',')]
                reg_values = [float(x.strip()) for x in self.reg_var.get().split(',')]
            except ValueError:
                messagebox.showerror("Error", "Invalid parameter values. Please enter comma-separated numbers.")
                return

            self.update_status(f"Running first optimization round for {model_name}...", start_progress=True)

            # Start optimization in a separate thread
            threading.Thread(target=self._run_first_optimization_thread,
                             args=(model_name, epochs, lr_values, reg_values,
                                   {"SVD": SVD, "SVDpp": SVDpp, "NMF": NMF, "KNNBasic": KNNBasic,
                                    "KNNWithMeans": KNNWithMeans})).start()

        def _run_first_optimization_thread(self, model_name, epochs, lr_values, reg_values, models):
            """Background thread for first optimization round"""
            try:
                # Get the model class from the dictionary
                algo_class = models[model_name]

                # Define parameter grid based on the model
                if model_name == "NMF":
                    # For NMF, use different parameter names - it uses reg_pu, reg_qi, reg_bu, reg_bi instead of reg_all
                    param_grid = {
                        'n_epochs': epochs,
                        'reg_pu': reg_values,  # regularization for user factors
                        'reg_qi': reg_values  # regularization for item factors
                    }
                elif model_name in ["KNNBasic", "KNNWithMeans"]:
                    # KNN models don't use these parameters
                    param_grid = {}
                    # For KNN models, we might want to optimize other parameters like k, sim_options, etc.
                    # This is a simplified version - you might want to add KNN-specific parameters
                    messagebox.showinfo("Info",
                                        f"{model_name} doesn't require hyperparameter optimization for the selected parameters.")
                    return
                else:
                    # For SVD and SVDpp, include all parameters
                    param_grid = {
                        'n_epochs': epochs,
                        'lr_all': lr_values,
                        'reg_all': reg_values
                    }

                # Create GridSearchCV
                gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=3)
                gs.fit(self.data)

                # Extract best params and scores
                best_params = gs.best_params['rmse']
                best_rmse = gs.best_score['rmse']
                best_mae = gs.best_score['mae']

                # Store results
                self.best_params['first_round'] = best_params
                self.current_model_name = model_name

                # Create results dict for visualization
                cv_results = gs.cv_results
                param_combos = []
                rmse_means = []
                mae_means = []

                for i, params in enumerate(cv_results['params']):
                    if model_name == "NMF":
                        param_str = f"epochs={params['n_epochs']}, reg={params['reg_all']}"
                    else:
                        param_str = f"epochs={params['n_epochs']}, lr={params.get('lr_all', 'N/A')}, reg={params['reg_all']}"
                    param_combos.append(param_str)
                    rmse_means.append(cv_results['mean_test_rmse'][i])
                    mae_means.append(cv_results['mean_test_mae'][i])

                # Update UI in main thread
                self.root.after(0, lambda: self._update_optimization_results(
                    'first_round', model_name, best_params, best_rmse, best_mae,
                    param_combos, rmse_means, mae_means))

            except Exception as e:
                error_msg = str(e)
                print(f"Error during optimization: {error_msg}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
                self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Optimization failed: {err}"))

        def _update_optimization_results(self, round_name, model_name, best_params, best_rmse, best_mae,
                                         param_combos, rmse_means, mae_means):
            """Update UI with optimization results"""
            # Update results text
            self.opt_results_text.config(state=tk.NORMAL)

            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            results_text = f"=== {round_name.replace('_', ' ').title()} ===\n"
            results_text += f"Time: {timestamp}\n"
            results_text += f"Model: {model_name}\n\n"
            results_text += f"Best Parameters:\n"

            for param, value in best_params.items():
                results_text += f"  {param}: {value}\n"

            results_text += f"\nBest RMSE: {best_rmse:.4f}\n"
            results_text += f"Best MAE: {best_mae:.4f}\n\n"

            self.opt_results_text.insert(tk.END, results_text)
            self.opt_results_text.see(tk.END)
            self.opt_results_text.config(state=tk.DISABLED)

            # Update visualization
            self._update_optimization_visualization(param_combos, rmse_means, mae_means)

            # Update status
            self.update_status(f"Optimization complete. Best RMSE: {best_rmse:.4f}")

        def _update_optimization_visualization(self, param_combos, rmse_means, mae_means):
            """Update visualization for optimization results"""
            # Clear previous chart
            for widget in self.opt_fig_frame.winfo_children():
                widget.destroy()

            # Create new figure with two subplots
            fig = Figure(figsize=(10, 6))

            # RMSE plot
            ax1 = fig.add_subplot(211)
            ax1.plot(range(len(param_combos)), rmse_means, 'o-', color=self.primary_color)
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE for Different Parameter Combinations')
            ax1.set_xticks([])  # Hide x ticks for this subplot

            # Find and mark best (minimum) RMSE
            best_idx = np.argmin(rmse_means)
            ax1.plot(best_idx, rmse_means[best_idx], 'o', color='red', markersize=10)
            ax1.annotate(f'Best: {rmse_means[best_idx]:.4f}',
                         xy=(best_idx, rmse_means[best_idx]),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

            # MAE plot
            ax2 = fig.add_subplot(212)
            ax2.plot(range(len(param_combos)), mae_means, 'o-', color=self.accent_color)
            ax2.set_ylabel('MAE')
            ax2.set_title('MAE for Different Parameter Combinations')

            # X axis with parameter combinations (shortened for readability)
            short_params = [p[:20] + '...' if len(p) > 20 else p for p in param_combos]
            ax2.set_xticks(range(len(param_combos)))
            ax2.set_xticklabels(short_params, rotation=45, ha='right')

            # Find and mark best (minimum) MAE
            best_idx = np.argmin(mae_means)
            ax2.plot(best_idx, mae_means[best_idx], 'o', color='red', markersize=10)
            ax2.annotate(f'Best: {mae_means[best_idx]:.4f}',
                         xy=(best_idx, mae_means[best_idx]),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))

            fig.tight_layout()

            # Add chart to frame
            canvas = FigureCanvasTkAgg(fig, master=self.opt_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_second_optimization(self):
        """Run second round of hyperparameter optimization with factors"""
        if 'first_round' not in self.best_params:
            messagebox.showwarning("Warning", "Please run first optimization round before proceeding.")
            return

        # Get model and parameters
        model_name = self.current_model_name
        best_params = self.best_params['first_round']

        # Parse factors range
        try:
            factors = [int(x.strip()) for x in self.factors_var.get().split(',')]
            # Validate factors
            if any(f <= 0 for f in factors):
                raise ValueError("Factors must be positive integers.")
        except ValueError:
            messagebox.showerror("Error", "Invalid factors values. Please enter comma-separated positive integers.")
            return

        self.update_status(f"Running second optimization round for {model_name}...", start_progress=True)

        # Start optimization in a separate thread
        threading.Thread(target=self._run_second_optimization_thread,
                         args=(model_name, best_params, factors)).start()

    def run_second_optimization(self):
        """Run second round of hyperparameter optimization with factors"""
        if 'first_round' not in self.best_params:
            messagebox.showwarning("Warning", "Please run first optimization round before proceeding.")
            return

        # Get model and parameters
        model_name = self.current_model_name
        best_params = self.best_params['first_round']

        # Parse factors range
        try:
            factors = [int(x.strip()) for x in self.factors_var.get().split(',')]
            # Validate factors
            if any(f <= 0 for f in factors):
                raise ValueError("Factors must be positive integers.")
        except ValueError:
            messagebox.showerror("Error", "Invalid factors values. Please enter comma-separated positive integers.")
            return

        self.update_status(f"Running second optimization round for {model_name}...", start_progress=True)

        # Start optimization in a separate thread
        threading.Thread(target=self._run_second_optimization_thread,
                         args=(model_name, best_params, factors)).start()

    def _run_second_optimization_thread(self, model_name, best_params, factors):
        """Background thread for second optimization round"""
        try:
            print(f"Starting second optimization for model: {model_name}")
            print(f"Best parameters from first round: {best_params}")
            print(f"Factors to test: {factors}")

            # Import necessary modules
            from surprise import SVD, SVDpp, NMF
            from surprise.model_selection import train_test_split
            from surprise import accuracy

            # Split data for evaluation
            trainset, testset = train_test_split(self.data, test_size=0.25, random_state=42)

            # Initialize variables to track best results
            best_rmse = float('inf')
            best_mae = float('inf')
            best_params_second = None

            # Store results for visualization
            param_combos = []
            rmse_means = []
            mae_means = []

            # Get parameters from first round based on model type
            n_epochs = best_params['n_epochs']

            # Handle model-specific parameters
            if model_name == "NMF":
                # NMF uses reg_pu and reg_qi instead of reg_all, and doesn't use lr_all
                reg_pu = best_params.get('reg_pu')
                reg_qi = best_params.get('reg_qi')
                lr_all = None  # NMF doesn't use lr_all
            elif model_name in ["SVD", "SVDpp"]:
                # SVD and SVDpp use both lr_all and reg_all
                lr_all = best_params.get('lr_all')
                reg_all = best_params.get('reg_all')
                reg_pu = None
                reg_qi = None
            else:
                # For other models that might not use these parameters
                lr_all = None
                reg_all = None
                reg_pu = None
                reg_qi = None

            # Try each factor value
            for n_factors in factors:
                try:
                    print(f"Testing n_factors={n_factors}")

                    # Create and train model based on model name
                    if model_name == "SVD":
                        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "SVDpp":
                        algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    elif model_name == "NMF":
                        # NMF doesn't use lr_all parameter and uses reg_pu, reg_qi instead of reg_all
                        algo = NMF(n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu, reg_qi=reg_qi)
                    else:
                        print(f"Unknown model: {model_name}, using SVD as fallback")
                        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

                    # Train model
                    algo.fit(trainset)

                    # Test model
                    predictions = algo.test(testset)
                    rmse = accuracy.rmse(predictions)
                    mae = accuracy.mae(predictions)

                    print(f"n_factors={n_factors}: RMSE={rmse:.4f}, MAE={mae:.4f}")

                    # Store results for this combination
                    if model_name == "NMF":
                        param_str = f"factors={n_factors}, epochs={n_epochs}, reg_pu={reg_pu}, reg_qi={reg_qi}"
                    else:
                        param_str = f"factors={n_factors}, epochs={n_epochs}, lr={lr_all}, reg={reg_all}"

                    param_combos.append(param_str)
                    rmse_means.append(rmse)
                    mae_means.append(mae)

                    # Update best parameters if this is better
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_mae = mae
                        best_params_second = {
                            'n_factors': n_factors,
                            'n_epochs': n_epochs
                        }

                        # Add model-specific parameters
                        if model_name == "NMF":
                            best_params_second['reg_pu'] = reg_pu
                            best_params_second['reg_qi'] = reg_qi
                        elif model_name in ["SVD", "SVDpp"]:
                            best_params_second['lr_all'] = lr_all
                            best_params_second['reg_all'] = reg_all

                except Exception as e:
                    print(f"Error with n_factors={n_factors}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add dummy results for failed combinations
                    param_str = f"factors={n_factors} (failed)"
                    param_combos.append(param_str)
                    rmse_means.append(None)
                    mae_means.append(None)

            if best_params_second is None:
                raise Exception("All factor values failed")

            print(f"Best parameters found: {best_params_second}")
            print(f"Best RMSE: {best_rmse}, Best MAE: {best_mae}")

            # Store results
            self.best_params['second_round'] = best_params_second

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'second_round', model_name, best_params_second, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            print(f"Error in second optimization thread: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Second optimization failed: {err}"))

    def train_final_model(self):
        """Train final model with best parameters"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get selected model name
        model_name = self.model_var.get()
        self.current_model_name = model_name

        # Check if we have optimized parameters
        use_optimized = False
        if 'second_round' in self.best_params and self.current_model_name in ["SVD", "SVDpp", "NMF"]:
            params = self.best_params['second_round']
            use_optimized = True
        elif 'first_round' in self.best_params:
            params = self.best_params['first_round']
            use_optimized = True

        self.update_status(f"Training {model_name} model...", start_progress=True)

        # Start training in a separate thread
        threading.Thread(target=self._train_final_model_thread,
                         args=(model_name, use_optimized, params if use_optimized else None)).start()

    def _train_final_model_thread(self, model_name, use_optimized, params=None):
        """Background thread for training final model"""
        try:
            # Create model - FIXED VERSION
            if model_name == "SVD":
                if use_optimized:
                    algo = SVD(n_factors=params.get('n_factors', 100),
                               n_epochs=params.get('n_epochs', 20),
                               lr_all=params.get('lr_all', 0.005),
                               reg_all=params.get('reg_all', 0.02))
                else:
                    algo = SVD()
            elif model_name == "SVDpp":
                if use_optimized:
                    algo = SVDpp(n_factors=params.get('n_factors', 100),
                                 n_epochs=params.get('n_epochs', 20),
                                 lr_all=params.get('lr_all', 0.005),
                                 reg_all=params.get('reg_all', 0.02))
                else:
                    algo = SVDpp()
            elif model_name == "NMF":
                if use_optimized:
                    # NMF doesn't use lr_all parameter and uses reg_pu, reg_qi instead of reg_all
                    algo = NMF(n_factors=params.get('n_factors', 15),
                               n_epochs=params.get('n_epochs', 50),
                               reg_pu=params.get('reg_pu', 0.06),
                               reg_qi=params.get('reg_qi', 0.06))
                else:
                    algo = NMF()
            elif model_name == "KNNBasic":
                algo = KNNBasic()
            elif model_name == "KNNWithMeans":
                algo = KNNWithMeans()
            elif model_name == "SlopeOne":
                algo = SlopeOne()
            elif model_name == "CoClustering":
                algo = CoClustering()
            else:
                algo = SVD()  # Default fallback

            # Split data and train
            trainset, testset = train_test_split(self.data, test_size=0.25)
            self.trainset = trainset
            self.testset = testset

            start_time = datetime.datetime.now()
            algo.fit(trainset)
            train_time = (datetime.datetime.now() - start_time).total_seconds()

            # Test on test set
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)

            # Store model and results
            self.model = algo

            # Store predictions for best/worst analysis
            test_predictions = []
            for uid, iid, true_r, est, _ in predictions:
                test_predictions.append((uid, iid, true_r, est, abs(true_r - est)))

            # Sort by error (ascending for best, descending for worst)
            best_predictions = sorted(test_predictions, key=lambda x: x[4])[:10]
            worst_predictions = sorted(test_predictions, key=lambda x: x[4], reverse=True)[:10]

            # Update UI in main thread
            self.root.after(0, lambda: self._update_final_model_ui(
                model_name, rmse, mae, train_time, use_optimized, best_predictions, worst_predictions))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Training failed: {err}"))
    def _update_final_model_ui(self, model_name, rmse, mae, train_time, use_optimized,
                               best_predictions, worst_predictions):
        """Update UI after final model training"""
        # Update model info
        info_text = f"Model: {model_name} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Train Time: {train_time:.2f}s"
        if use_optimized:
            info_text += " | Using optimized parameters"

        self.model_info_label.config(text=info_text)

        # Update best/worst prediction trees
        # Clear previous entries
        for tree in [self.best_tree, self.worst_tree]:
            for item in tree.get_children():
                tree.delete(item)

        # Insert best predictions
        for uid, iid, true_r, est, error in best_predictions:
            self.best_tree.insert('', 'end', values=(uid, iid, f"{true_r:.1f}", f"{est:.2f}", f"{error:.4f}"))

        # Insert worst predictions
        for uid, iid, true_r, est, error in worst_predictions:
            self.worst_tree.insert('', 'end', values=(uid, iid, f"{true_r:.1f}", f"{est:.2f}", f"{error:.4f}"))

        self.update_status(f"{model_name} model trained successfully. RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    def get_recommendations(self):
        """Get recommendations for selected user"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first.")
            return

        # Get selected user
        user_id = self.user_var.get()
        if not user_id:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        self.update_status(f"Generating recommendations for user {user_id}...")

        # Clear previous recommendations
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)

        try:
            # Get all items this user hasn't rated
            user_items = set(self.df[self.df['user_id'] == user_id]['item_id'])
            all_items = set(self.df['item_id'].unique())
            items_to_predict = list(all_items - user_items)

            # If too many items, sample a reasonable number
            if len(items_to_predict) > 1000:
                items_to_predict = random.sample(items_to_predict, 1000)

            # Generate predictions
            predictions = []
            for item_id in items_to_predict:
                est = self.model.predict(user_id, item_id).est
                predictions.append((item_id, est))

            # Sort by predicted rating (descending)
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Store for potential export
            self.recommendations = predictions[:20]

            # Display top recommendations
            for i, (item_id, rating) in enumerate(predictions[:20]):
                self.rec_tree.insert('', 'end', values=(item_id, f"{rating:.2f}"))

            self.update_status(f"Generated {len(predictions)} recommendations for user {user_id}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {str(e)}")
            self.update_status("Error generating recommendations")

    def find_similar_items(self):
        """Find similar items to the selected item"""
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first.")
            return

        # Get selected item
        item_id = self.item_var.get()
        if not item_id:
            messagebox.showwarning("Warning", "Please select an item.")
            return

        # This only works for certain model types that have item factors
        if self.current_model_name not in ["SVD", "SVDpp", "NMF"]:
            messagebox.showinfo("Information",
                                f"The {self.current_model_name} model doesn't support item similarity computation. Please use SVD, SVD++, or NMF.")
            return

        self.update_status(f"Finding items similar to {item_id}...")

        # Clear previous similar items
        if hasattr(self, 'similar_tree'):
            for item in self.similar_tree.get_children():
                self.similar_tree.delete(item)

        try:
            # Use model to find similar items
            # This is a simplified approach - in a real implementation,
            # we would use the item factors from the model

            # For matrix factorization models, we can use the item factors
            if hasattr(self.model, 'qi') and item_id in self.model.trainset.to_inner_iid:
                inner_id = self.model.trainset.to_inner_iid(item_id)
                item_factors = self.model.qi[inner_id]

                # Calculate similarity to all other items
                similarities = []
                for other_item in self.df['item_id'].unique():
                    if other_item != item_id and other_item in self.model.trainset.to_inner_iid:
                        other_inner_id = self.model.trainset.to_inner_iid(other_item)
                        other_factors = self.model.qi[other_inner_id]

                        # Cosine similarity
                        sim = np.dot(item_factors, other_factors) / (
                                np.linalg.norm(item_factors) * np.linalg.norm(other_factors))
                        similarities.append((other_item, sim))

                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)

                # Display top similar items
                for i, (other_item, sim) in enumerate(similarities[:10]):
                    self.similar_tree.insert('', 'end', values=(other_item, f"{sim:.4f}"))

                self.update_status(f"Found similar items to {item_id}")
            else:
                # Fallback to a simple collaborative approach
                # Get users who rated this item highly
                item_ratings = self.df[self.df['item_id'] == item_id]
                if len(item_ratings) == 0:
                    messagebox.showinfo("Information", f"No ratings found for item {item_id}")
                    return

                high_ratings = item_ratings[item_ratings['rating'] > item_ratings['rating'].mean()]
                if len(high_ratings) == 0:
                    high_ratings = item_ratings  # Use all ratings if none are above average

                users = high_ratings['user_id'].unique()

                # Find other items these users rated highly
                other_items = {}
                for user in users:
                    user_ratings = self.df[(self.df['user_id'] == user) & (self.df['item_id'] != item_id)]
                    for _, row in user_ratings.iterrows():
                        other_item = row['item_id']
                        rating = row['rating']
                        if other_item not in other_items:
                            other_items[other_item] = []
                        other_items[other_item].append(rating)

                # Calculate average rating for each item
                similarities = []
                for other_item, ratings in other_items.items():
                    avg_rating = sum(ratings) / len(ratings)
                    count = len(ratings)  # How many users rated both items
                    similarities.append((other_item, avg_rating, count))

                # Sort by count (more shared users) and then by average rating
                similarities.sort(key=lambda x: (x[2], x[1]), reverse=True)

                # Display top similar items
                for i, (other_item, avg_rating, count) in enumerate(similarities[:10]):
                    similarity_score = avg_rating / 5.0  # Normalize to 0-1 scale
                    self.similar_tree.insert('', 'end', values=(other_item, f"{similarity_score:.4f} ({count} users)"))

                self.update_status(f"Found similar items to {item_id} using collaborative approach")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to find similar items: {str(e)}")
            self.update_status("Error finding similar items")

    def run_evaluation(self):
        """Run comprehensive model evaluation"""
        if self.data is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        # Get parameters
        model_name = self.eval_model_var.get()
        cv_folds = self.cv_folds_var.get()
        test_size = self.test_size_var.get()
        seed = self.eval_seed_var.get()
        k = self.k_var.get()

        # Get selected metrics
        metrics = {key: var.get() for key, var in self.metric_vars.items()}

        if not any(metrics.values()):
            messagebox.showwarning("Warning", "Please select at least one metric to evaluate.")
            return

        self.update_status(f"Evaluating {model_name}...", start_progress=True)

        # Start evaluation in a separate thread
        threading.Thread(target=self._run_evaluation_thread,
                         args=(model_name, cv_folds, test_size, seed, k, metrics)).start()

    def generate_visualization(self):
        """Generate selected visualization"""
        if self.data is None or self.df is None:
            messagebox.showwarning("Warning", "No data available. Please generate or import data first.")
            return

        chart_type = self.chart_type_var.get()
        sample_size = self.sample_size_var.get()
        color_scheme = self.color_scheme_var.get()
        fig_width = self.fig_width_var.get()
        fig_height = self.fig_height_var.get()

        self.update_status(f"Generating {chart_type} visualization...", start_progress=True)

        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Create function-specific visualizations based on chart type
        try:
            # Create new figure
            fig = Figure(figsize=(fig_width, fig_height))

            if chart_type == "distribution":
                self._create_distribution_chart(fig, color_scheme)
            elif chart_type == "user_activity":
                self._create_user_activity_chart(fig, sample_size, color_scheme)
            elif chart_type == "item_popularity":
                self._create_item_popularity_chart(fig, sample_size, color_scheme)
            elif chart_type == "heatmap":
                self._create_rating_heatmap(fig, sample_size, color_scheme)
            elif chart_type == "user_similarity":
                self._create_user_similarity_chart(fig, sample_size, color_scheme)
            elif chart_type == "item_similarity":
                self._create_item_similarity_chart(fig, sample_size, color_scheme)

            # Add chart to frame
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Store current figure for export
            self.current_figure = fig

            self.update_status(f"{chart_type.replace('_', ' ').title()} visualization generated")

        except Exception as e:
            self.update_status("Ready")
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")

    def on_exit(self):
        """Handle application exit"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            # Save config before exiting
            self.save_config()

            # Clean up any resources
            for thread in self.active_threads:
                if thread.is_alive():
                    # Can't forcibly terminate threads in Python, but we can set a flag
                    # if the threads check for it
                    pass

            self.root.destroy()

    def show_documentation(self):
        """Show documentation for the application"""
        # Create a new top-level window for documentation
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("800x600")
        doc_window.minsize(600, 400)

        # Add a frame with scrollable text
        frame = ttk.Frame(doc_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a scrolled text widget
        doc_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Helvetica", 11))
        doc_text.pack(fill=tk.BOTH, expand=True)

        # Documentation content
        documentation = """
        # Enhanced E-Commerce Recommendation System Documentation

        ## Overview
        This application provides tools for building and evaluating recommendation systems for e-commerce data.
        It implements several collaborative filtering algorithms with a focus on matrix factorization techniques.

        ## Data Tab
        - **Generate Data**: Create synthetic e-commerce data with configurable parameters
        - **Import Data**: Load your own data from CSV or Excel files
        - **Data Statistics**: View summary statistics of the loaded dataset

        ## Algorithm Tab
        - **Compare Algorithms**: Evaluate multiple recommendation algorithms side by side
        - **Performance Metrics**: Compare RMSE, MAE, and training/testing times

        ## Optimization Tab
        - **Hyperparameter Tuning**: Optimize algorithm parameters for best performance
        - **Two-Stage Optimization**: First optimize learning parameters, then latent factors

        ## Recommendation Tab
        - **Train Model**: Train a model with optimized parameters
        - **Get Recommendations**: Generate personalized recommendations for users
        - **Find Similar Items**: Discover items similar to a selected item

        ## Evaluation Tab
        - **Comprehensive Metrics**: Evaluate models using various metrics
        - **Cross-Validation**: Perform k-fold cross-validation
        - **Precision-Recall Analysis**: Analyze ranking performance

        ## Visualization Tab
        - **Data Visualizations**: Explore data distributions and patterns
        - **User/Item Similarity**: Visualize latent factor spaces
        - **Custom Charts**: Configure chart parameters

        ## Settings Tab
        - **Application Preferences**: Customize the application
        - **Theme Settings**: Choose between light and dark themes

        ## Tips for Best Results
        1. Start with data exploration to understand your dataset
        2. Compare multiple algorithms to find the best baseline
        3. Use hyperparameter optimization to improve performance
        4. Evaluate with metrics relevant to your business goals
        5. Visualize results to gain insights

        ## Supported Algorithms
        - SVD (Singular Value Decomposition)
        - SVD++ (Enhanced SVD with implicit feedback)
        - NMF (Non-negative Matrix Factorization)
        - KNN (K-Nearest Neighbors variants)
        - SlopeOne
        - CoClustering

        ## File Formats
        The application supports CSV and Excel files with the following columns:
        - user_id: Identifier for users
        - item_id: Identifier for items
        - rating: Numerical rating value

        If your data uses different column names, the application will prompt you to map them.
        """

        # Insert documentation text
        doc_text.insert(tk.END, documentation)
        doc_text.config(state=tk.DISABLED)  # Make read-only

        # Add a close button
        close_button = ttk.Button(doc_window, text="Close", command=doc_window.destroy)
        close_button.pack(pady=10)

    def show_about(self):
        """Show information about the application"""
        about_text = """
        Enhanced E-Commerce Recommendation System
        Version 1.0

        A comprehensive tool for building and evaluating 
        recommendation systems for e-commerce applications.

        Features:
        • Multiple recommendation algorithms
        • Hyperparameter optimization
        • Comprehensive evaluation metrics
        • Interactive visualizations
        • User and item similarity analysis

        Developed with Python using Surprise, Pandas, NumPy, 
        Matplotlib, and Tkinter.

        © 2023 All Rights Reserved
        """

        messagebox.showinfo("About", about_text)

# Main function to run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedSVDRecommenderApp(root)
    root.mainloop()