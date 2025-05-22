import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import threading
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from skimage import exposure
import seaborn as sns


class PengenalWajahSVDLanjutan:
    """
    Kelas lanjutan untuk implementasi pengenalan wajah menggunakan SVD
    dengan fitur-fitur canggih dan antarmuka GUI
    """

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.mean_face = None
        self.components = None
        self.face_projected = None
        self.labels = None
        self.label_names = {}
        self.image_shape = None
        self.model_trained = False
        self.training_history = []

    def preprocessing_gambar(self, gambar, metode='clahe'):
        """Preprocessing gambar dengan berbagai metode"""
        if len(gambar.shape) == 3:
            gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

        # Normalisasi ukuran
        gambar = cv2.resize(gambar, (100, 100))

        # Normalisasi intensitas
        gambar = gambar.astype(np.float32) / 255.0

        if metode == 'hist_eq':
            gambar = exposure.equalize_hist(gambar)
        elif metode == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gambar = clahe.apply((gambar * 255).astype(np.uint8)) / 255.0
        elif metode == 'gamma':
            gambar = exposure.adjust_gamma(gambar, gamma=1.2)
        elif metode == 'adaptive':
            # Metode adaptif berdasarkan histogram
            hist = np.histogram(gambar, bins=256)[0]
            if np.mean(gambar) < 0.4:  # Gambar gelap
                gambar = exposure.adjust_gamma(gambar, gamma=0.7)
            elif np.mean(gambar) > 0.7:  # Gambar terang
                gambar = exposure.adjust_gamma(gambar, gamma=1.3)
            gambar = exposure.equalize_adapthist(gambar)

        return gambar

    def deteksi_wajah(self, gambar):
        """Deteksi wajah menggunakan Haar Cascade"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if len(gambar.shape) == 3:
            gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
        else:
            gray = gambar

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Ambil wajah terbesar
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            return gray[y:y + h, x:x + w], largest_face

        return None, None

    def latih_model(self, X, y, label_names=None, metode_preprocessing='clahe'):
        """Latih model SVD dengan data training"""
        try:
            n_samples = len(X)
            if n_samples == 0:
                raise ValueError("Tidak ada data training")

            # Preprocessing semua gambar
            X_processed = []
            for img in X:
                processed_img = self.preprocessing_gambar(img, metode_preprocessing)
                X_processed.append(processed_img.flatten())

            X_processed = np.array(X_processed)
            self.image_shape = (100, 100)

            # Hitung mean face
            self.mean_face = np.mean(X_processed, axis=0)

            # Center data
            X_centered = X_processed - self.mean_face

            # SVD
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Ambil komponen utama
            self.components = Vt[:self.n_components]

            # Proyeksi data training
            self.face_projected = np.dot(X_centered, self.components.T)
            self.labels = np.array(y)

            if label_names:
                self.label_names = label_names
            else:
                self.label_names = {i: f"Orang_{i}" for i in np.unique(y)}

            self.model_trained = True

            # Simpan riwayat training
            training_info = {
                'timestamp': datetime.now().isoformat(),
                'n_samples': n_samples,
                'n_components': self.n_components,
                'n_identities': len(np.unique(y)),
                'preprocessing': metode_preprocessing
            }
            self.training_history.append(training_info)

            return True, "Model berhasil dilatih!"

        except Exception as e:
            return False, f"Error saat melatih model: {str(e)}"

    def prediksi(self, gambar, metode_preprocessing='clahe', threshold=0.8):
        """Prediksi identitas wajah"""
        if not self.model_trained:
            return None, None, "Model belum dilatih"

        try:
            # Preprocessing
            processed_img = self.preprocessing_gambar(gambar, metode_preprocessing)
            img_flat = processed_img.flatten()

            # Center dan proyeksi
            img_centered = img_flat - self.mean_face
            img_projected = np.dot(img_centered, self.components.T)

            # Hitung jarak ke semua wajah training
            distances = np.linalg.norm(self.face_projected - img_projected, axis=1)
            min_distance = np.min(distances)
            min_idx = np.argmin(distances)

            # Confidence berdasarkan jarak
            confidence = max(0, 1 - (min_distance / threshold))

            if confidence > 0.3:  # Threshold minimum confidence
                predicted_label = self.labels[min_idx]
                predicted_name = self.label_names.get(predicted_label, f"Unknown_{predicted_label}")
                return predicted_label, predicted_name, confidence
            else:
                return None, "Unknown", confidence

        except Exception as e:
            return None, None, f"Error saat prediksi: {str(e)}"

    def evaluasi_model(self, X_test, y_test, metode_preprocessing='clahe'):
        """Evaluasi performa model"""
        if not self.model_trained:
            return None

        y_pred = []
        confidences = []

        for img, true_label in zip(X_test, y_test):
            pred_label, pred_name, confidence = self.prediksi(img, metode_preprocessing)
            y_pred.append(pred_label if pred_label is not None else -1)
            confidences.append(confidence)

        # Hitung metrik
        y_pred = np.array(y_pred)
        mask = y_pred != -1  # Hanya hitung yang berhasil diprediksi

        if np.sum(mask) > 0:
            accuracy = accuracy_score(y_test[mask], y_pred[mask])
            cm = confusion_matrix(y_test[mask], y_pred[mask])
            avg_confidence = np.mean([c for c, m in zip(confidences, mask) if m])
        else:
            accuracy = 0
            cm = np.array([[0]])
            avg_confidence = 0

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'avg_confidence': avg_confidence,
            'predictions': y_pred,
            'confidences': confidences
        }


class AplikasiPengenalWajahGUI:
    """
    Aplikasi GUI untuk pengenalan wajah menggunakan SVD
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Pengenalan Wajah SVD Lanjutan")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Inisialisasi model
        self.model = PengenalWajahSVDLanjutan()
        self.dataset = {'images': [], 'labels': [], 'names': {}}
        self.camera = None
        self.camera_running = False

        # Setup GUI
        self.setup_gui()
        self.setup_styles()

    def setup_styles(self):
        """Setup styling untuk GUI"""
        style = ttk.Style()
        style.theme_use('clam')

        # Konfigurasi warna
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')

    def setup_gui(self):
        """Setup antarmuka pengguna"""
        # Frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Judul
        title_label = ttk.Label(main_frame, text="Sistem Pengenalan Wajah SVD Lanjutan",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Notebook untuk tab
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Dataset Management
        self.setup_dataset_tab()

        # Tab 2: Training
        self.setup_training_tab()

        # Tab 3: Recognition
        self.setup_recognition_tab()

        # Tab 4: Analysis
        self.setup_analysis_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Siap - Silakan muat dataset atau ambil foto untuk memulai")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, style='Info.TLabel')
        status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def setup_dataset_tab(self):
        """Setup tab untuk manajemen dataset"""
        dataset_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(dataset_frame, text="Dataset")

        # Frame kiri - kontrol
        control_frame = ttk.LabelFrame(dataset_frame, text="Kontrol Dataset", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Tombol muat dataset
        ttk.Button(control_frame, text="Muat Dataset dari Folder",
                   command=self.muat_dataset).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)

        # Tombol buka kamera
        self.camera_btn = ttk.Button(control_frame, text="Buka Kamera",
                                     command=self.toggle_camera)
        self.camera_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        # Tombol ambil foto
        self.capture_btn = ttk.Button(control_frame, text="Ambil Foto",
                                      command=self.ambil_foto, state='disabled')
        self.capture_btn.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)

        # Input nama untuk foto baru
        ttk.Label(control_frame, text="Nama untuk foto baru:").grid(row=3, column=0, sticky=tk.W, pady=(10, 2))
        self.nama_entry = ttk.Entry(control_frame)
        self.nama_entry.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=2)

        # Tombol hapus dataset
        ttk.Button(control_frame, text="Hapus Semua Dataset",
                   command=self.hapus_dataset).grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 2))

        # Info dataset
        info_frame = ttk.LabelFrame(control_frame, text="Informasi Dataset", padding="5")
        info_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.dataset_info = tk.Text(info_frame, height=8, width=30)
        self.dataset_info.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Frame kanan - preview
        preview_frame = ttk.LabelFrame(dataset_frame, text="Preview", padding="10")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Canvas untuk kamera/preview
        self.canvas = tk.Canvas(preview_frame, width=400, height=300, bg='black')
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Configure grid weights
        dataset_frame.columnconfigure(1, weight=1)
        dataset_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

    def setup_training_tab(self):
        """Setup tab untuk training model"""
        training_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(training_frame, text="Training")

        # Frame kontrol training
        control_frame = ttk.LabelFrame(training_frame, text="Pengaturan Training", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Pengaturan komponen SVD
        ttk.Label(control_frame, text="Jumlah Komponen SVD:").grid(row=0, column=0, sticky=tk.W)
        self.components_var = tk.IntVar(value=50)
        components_scale = ttk.Scale(control_frame, from_=10, to=100, variable=self.components_var,
                                     orient=tk.HORIZONTAL)
        components_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.components_label = ttk.Label(control_frame, text="50")
        self.components_label.grid(row=0, column=2, padx=(5, 0))
        components_scale.configure(command=self.update_components_label)

        # Metode preprocessing
        ttk.Label(control_frame, text="Metode Preprocessing:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.preprocessing_var = tk.StringVar(value="clahe")
        preprocessing_combo = ttk.Combobox(control_frame, textvariable=self.preprocessing_var,
                                           values=["clahe", "hist_eq", "gamma", "adaptive"], state="readonly")
        preprocessing_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))

        # Tombol mulai training
        ttk.Button(control_frame, text="Mulai Training",
                   command=self.mulai_training).grid(row=2, column=0, columnspan=3, pady=(20, 0))

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # Frame hasil training
        result_frame = ttk.LabelFrame(training_frame, text="Hasil Training", padding="10")
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.training_result = tk.Text(result_frame, height=15)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.training_result.yview)
        self.training_result.configure(yscrollcommand=scrollbar.set)
        self.training_result.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid weights
        training_frame.columnconfigure(0, weight=1)
        training_frame.rowconfigure(1, weight=1)
        control_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

    def setup_recognition_tab(self):
        """Setup tab untuk pengenalan wajah"""
        recognition_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(recognition_frame, text="Pengenalan")

        # Frame kiri - kontrol
        control_frame = ttk.LabelFrame(recognition_frame, text="Kontrol Pengenalan", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Tombol untuk pengenalan
        ttk.Button(control_frame, text="Pilih Gambar untuk Dikenali",
                   command=self.pilih_gambar_pengenalan).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)

        ttk.Button(control_frame, text="Gunakan Kamera untuk Pengenalan",
                   command=self.mulai_pengenalan_kamera).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        # Threshold confidence
        ttk.Label(control_frame, text="Threshold Confidence:").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.threshold_var,
                                    orient=tk.HORIZONTAL)
        threshold_scale.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)

        # Frame hasil pengenalan
        result_frame = ttk.LabelFrame(control_frame, text="Hasil Pengenalan", padding="5")
        result_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        self.recognition_result = tk.Text(result_frame, height=10, width=30)
        self.recognition_result.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Frame kanan - gambar
        image_frame = ttk.LabelFrame(recognition_frame, text="Gambar", padding="10")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.recognition_canvas = tk.Canvas(image_frame, width=400, height=400, bg='white')
        self.recognition_canvas.grid(row=0, column=0)

        # Configure grid weights
        recognition_frame.columnconfigure(1, weight=1)
        recognition_frame.rowconfigure(0, weight=1)
        control_frame.columnconfigure(0, weight=1)

    def setup_analysis_tab(self):
        """Setup tab untuk analisis dan visualisasi"""
        analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(analysis_frame, text="Analisis")

        # Tombol analisis
        button_frame = ttk.Frame(analysis_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(button_frame, text="Tampilkan Eigenfaces",
                   command=self.tampilkan_eigenfaces).grid(row=0, column=0, padx=(0, 10))

        ttk.Button(button_frame, text="Evaluasi Model",
                   command=self.evaluasi_model).grid(row=0, column=1, padx=(0, 10))

        ttk.Button(button_frame, text="Analisis Komponen",
                   command=self.analisis_komponen).grid(row=0, column=2)

        # Frame untuk plot
        self.plot_frame = ttk.Frame(analysis_frame)
        self.plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(1, weight=1)

    def update_components_label(self, value):
        """Update label jumlah komponen"""
        self.components_label.config(text=str(int(float(value))))

    def muat_dataset(self):
        """Muat dataset dari folder"""
        folder_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if not folder_path:
            return

        try:
            self.dataset = {'images': [], 'labels': [], 'names': {}}
            label_counter = 0

            for person_folder in os.listdir(folder_path):
                person_path = os.path.join(folder_path, person_folder)
                if os.path.isdir(person_path):
                    self.dataset['names'][label_counter] = person_folder

                    for img_file in os.listdir(person_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(person_path, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                # Deteksi wajah
                                face, bbox = self.model.deteksi_wajah(img)
                                if face is not None:
                                    self.dataset['images'].append(face)
                                    self.dataset['labels'].append(label_counter)
                                else:
                                    # Jika tidak ada wajah terdeteksi, gunakan gambar asli
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    self.dataset['images'].append(gray)
                                    self.dataset['labels'].append(label_counter)

                    label_counter += 1

            self.update_dataset_info()
            self.status_var.set(
                f"Dataset dimuat: {len(self.dataset['images'])} gambar dari {len(self.dataset['names'])} orang")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat dataset: {str(e)}")

    def toggle_camera(self):
        """Toggle kamera on/off"""
        if not self.camera_running:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera_running = True
                self.camera_btn.config(text="Tutup Kamera")
                self.capture_btn.config(state='normal')
                self.update_camera()
            else:
                messagebox.showerror("Error", "Tidak dapat membuka kamera")
        else:
            self.camera_running = False
            if self.camera:
                self.camera.release()
            self.camera_btn.config(text="Buka Kamera")
            self.capture_btn.config(state='disabled')
            self.canvas.delete("all")

    def update_camera(self):
        """Update tampilan kamera"""
        if self.camera_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                # Flip horizontal untuk efek mirror
                frame = cv2.flip(frame, 1)

                # Deteksi wajah dan gambar bounding box
                face, bbox = self.model.deteksi_wajah(frame)
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Wajah Terdeteksi", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Konversi ke format Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((400, 300))
                frame_tk = ImageTk.PhotoImage(frame_pil)

                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(200, 150, image=frame_tk)
                self.canvas.image = frame_tk

            # Schedule next update
            self.root.after(30, self.update_camera)

    def ambil_foto(self):
        """Ambil foto dari kamera"""
        if not self.camera_running or not self.camera:
            messagebox.showerror("Error", "Kamera tidak aktif")
            return

        nama = self.nama_entry.get().strip()
        if not nama:
            messagebox.showerror("Error", "Masukkan nama terlebih dahulu")
            return

        ret, frame = self.camera.read()
        if ret:
            # Deteksi wajah
            face, bbox = self.model.deteksi_wajah(frame)
            if face is not None:
                # Cari atau buat label untuk nama ini
                label = None
                for existing_label, existing_name in self.dataset['names'].items():
                    if existing_name == nama:
                        label = existing_label
                        break

                if label is None:
                    label = len(self.dataset['names'])
                    self.dataset['names'][label] = nama

                # Tambahkan ke dataset
                self.dataset['images'].append(face)
                self.dataset['labels'].append(label)

                self.update_dataset_info()
                self.status_var.set(f"Foto {nama} berhasil ditambahkan ke dataset")

                # Kosongkan entry nama
                self.nama_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", "Tidak ada wajah terdeteksi")
        else:
            messagebox.showerror("Error", "Gagal mengambil foto dari kamera")

    def hapus_dataset(self):
        """Hapus semua data dalam dataset"""
        if messagebox.askyesno("Konfirmasi", "Apakah Anda yakin ingin menghapus semua dataset?"):
            self.dataset = {'images': [], 'labels': [], 'names': {}}
            self.update_dataset_info()
            self.status_var.set("Dataset dihapus")

    def update_dataset_info(self):
        """Update informasi dataset"""
        self.dataset_info.delete(1.0, tk.END)

        if not self.dataset['images']:
            self.dataset_info.insert(tk.END, "Dataset kosong\n")
            return

        info = f"Total gambar: {len(self.dataset['images'])}\n"
        info += f"Jumlah orang: {len(self.dataset['names'])}\n\n"

        # Hitung distribusi per orang
        label_counts = {}
        for label in self.dataset['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1

        info += "Distribusi per orang:\n"
        for label, count in label_counts.items():
            name = self.dataset['names'].get(label, f"Unknown_{label}")
            info += f"- {name}: {count} gambar\n"

        self.dataset_info.insert(tk.END, info)

    def mulai_training(self):
        """Mulai proses training model"""
        if not self.dataset['images']:
            messagebox.showerror("Error", "Dataset kosong. Muat dataset terlebih dahulu.")
            return

        def training_thread():
            try:
                # Update progress
                self.progress_var.set(10)
                self.root.update()

                # Setup model dengan parameter yang dipilih
                n_components = self.components_var.get()
                preprocessing_method = self.preprocessing_var.get()

                self.model = PengenalWajahSVDLanjutan(n_components=n_components)

                self.progress_var.set(30)
                self.root.update()

                # Training
                success, message = self.model.latih_model(
                    self.dataset['images'],
                    self.dataset['labels'],
                    self.dataset['names'],
                    preprocessing_method
                )

                self.progress_var.set(70)
                self.root.update()

                if success:
                    # Evaluasi model dengan train/test split
                    if len(self.dataset['images']) > 1:
                        X_train, X_test, y_train, y_test = train_test_split(
                            self.dataset['images'],
                            self.dataset['labels'],
                            test_size=0.3,
                            random_state=42,
                            stratify=self.dataset['labels'] if len(np.unique(self.dataset['labels'])) > 1 else None
                        )

                        # Evaluasi
                        eval_results = self.model.evaluasi_model(X_test, y_test, preprocessing_method)

                        self.progress_var.set(90)
                        self.root.update()

                        # Tampilkan hasil
                        result_text = f"=== HASIL TRAINING ===\n\n"
                        result_text += f"Status: {message}\n"
                        result_text += f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        result_text += f"Jumlah komponen SVD: {n_components}\n"
                        result_text += f"Metode preprocessing: {preprocessing_method}\n"
                        result_text += f"Total data training: {len(X_train)}\n"
                        result_text += f"Total data testing: {len(X_test)}\n"
                        result_text += f"Jumlah identitas: {len(self.dataset['names'])}\n\n"

                        if eval_results:
                            result_text += f"=== EVALUASI MODEL ===\n"
                            result_text += f"Akurasi: {eval_results['accuracy']:.4f} ({eval_results['accuracy'] * 100:.2f}%)\n"
                            result_text += f"Confidence rata-rata: {eval_results['avg_confidence']:.4f}\n\n"

                            # Confusion matrix info
                            cm = eval_results['confusion_matrix']
                            result_text += f"Confusion Matrix:\n"
                            for i, name in self.dataset['names'].items():
                                if i < len(cm):
                                    true_pos = cm[i, i] if i < cm.shape[1] else 0
                                    total_true = np.sum(cm[i, :]) if i < cm.shape[0] else 0
                                    precision = true_pos / total_true if total_true > 0 else 0
                                    result_text += f"- {name}: {precision:.3f} precision\n"
                    else:
                        result_text = f"=== HASIL TRAINING ===\n\n"
                        result_text += f"Status: {message}\n"
                        result_text += f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        result_text += f"Jumlah komponen SVD: {n_components}\n"
                        result_text += f"Metode preprocessing: {preprocessing_method}\n"
                        result_text += f"Total data: {len(self.dataset['images'])}\n"
                        result_text += f"Jumlah identitas: {len(self.dataset['names'])}\n\n"
                        result_text += "Catatan: Dataset terlalu kecil untuk evaluasi train/test split\n"

                    self.progress_var.set(100)
                    self.status_var.set("Training selesai - Model siap digunakan")

                else:
                    result_text = f"=== TRAINING GAGAL ===\n\n"
                    result_text += f"Error: {message}\n"
                    result_text += f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    self.status_var.set("Training gagal")

                # Update UI
                self.training_result.delete(1.0, tk.END)
                self.training_result.insert(tk.END, result_text)

            except Exception as e:
                error_text = f"=== ERROR TRAINING ===\n\n"
                error_text += f"Error: {str(e)}\n"
                error_text += f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

                self.training_result.delete(1.0, tk.END)
                self.training_result.insert(tk.END, error_text)
                self.status_var.set("Training error")

            finally:
                self.progress_var.set(0)

        # Jalankan training di thread terpisah
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

    def pilih_gambar_pengenalan(self):
        """Pilih gambar untuk dikenali"""
        if not self.model.model_trained:
            messagebox.showerror("Error", "Model belum dilatih. Lakukan training terlebih dahulu.")
            return

        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            try:
                # Baca dan proses gambar
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("Error", "Gagal membaca gambar")
                    return

                # Deteksi wajah
                face, bbox = self.model.deteksi_wajah(img)

                if face is not None:
                    # Gunakan wajah yang terdeteksi
                    processed_img = face

                    # Gambar bounding box pada gambar asli untuk visualisasi
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, "Wajah Terdeteksi", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Jika tidak ada wajah terdeteksi, gunakan gambar asli
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.putText(img, "Tidak ada wajah terdeteksi", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Prediksi
                threshold = self.threshold_var.get()
                pred_label, pred_name, confidence = self.model.prediksi(
                    processed_img,
                    self.preprocessing_var.get(),
                    threshold
                )

                # Tampilkan hasil
                result_text = f"=== HASIL PENGENALAN ===\n\n"
                result_text += f"File: {os.path.basename(file_path)}\n"
                result_text += f"Waktu: {datetime.now().strftime('%H:%M:%S')}\n\n"

                if pred_label is not None:
                    result_text += f"Identitas: {pred_name}\n"
                    result_text += f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)\n"
                    result_text += f"Status: {'DIKENALI' if confidence > 0.5 else 'RAGU-RAGU'}\n"

                    # Tambahkan label pada gambar
                    label_text = f"{pred_name} ({confidence * 100:.1f}%)"
                    cv2.putText(img, label_text, (10, img.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    result_text += f"Identitas: Tidak Dikenali\n"
                    result_text += f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)\n"
                    result_text += f"Status: TIDAK DIKENALI\n"

                    cv2.putText(img, "TIDAK DIKENALI", (10, img.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Update UI
                self.recognition_result.delete(1.0, tk.END)
                self.recognition_result.insert(tk.END, result_text)

                # Tampilkan gambar
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                # Resize untuk fit canvas
                canvas_width = 400
                canvas_height = 400
                img_ratio = img_pil.width / img_pil.height

                if img_ratio > 1:  # Landscape
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:  # Portrait
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)

                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img_pil)

                self.recognition_canvas.delete("all")
                self.recognition_canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
                self.recognition_canvas.image = img_tk

                self.status_var.set(f"Pengenalan selesai: {pred_name if pred_name else 'Tidak dikenali'}")

            except Exception as e:
                messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}")

    def mulai_pengenalan_kamera(self):
        """Mulai pengenalan real-time menggunakan kamera"""
        if not self.model.model_trained:
            messagebox.showerror("Error", "Model belum dilatih. Lakukan training terlebih dahulu.")
            return

        # Buka window baru untuk real-time recognition
        self.realtime_window = tk.Toplevel(self.root)
        self.realtime_window.title("Pengenalan Wajah Real-time")
        self.realtime_window.geometry("800x600")

        # Canvas untuk video
        video_canvas = tk.Canvas(self.realtime_window, width=640, height=480, bg='black')
        video_canvas.pack(pady=10)

        # Frame kontrol
        control_frame = ttk.Frame(self.realtime_window)
        control_frame.pack(pady=10)

        # Tombol stop
        stop_btn = ttk.Button(control_frame, text="Stop",
                              command=lambda: self.stop_realtime_recognition())
        stop_btn.pack()

        # Mulai kamera untuk real-time recognition
        self.realtime_camera = cv2.VideoCapture(0)
        self.realtime_running = True

        def update_realtime():
            if self.realtime_running and self.realtime_camera:
                ret, frame = self.realtime_camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)  # Mirror effect

                    # Deteksi wajah
                    face, bbox = self.model.deteksi_wajah(frame)

                    if face is not None and bbox is not None:
                        x, y, w, h = bbox

                        # Prediksi
                        pred_label, pred_name, confidence = self.model.prediksi(
                            face,
                            self.preprocessing_var.get(),
                            self.threshold_var.get()
                        )

                        # Gambar bounding box dan label
                        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        if pred_label is not None:
                            label_text = f"{pred_name} ({confidence * 100:.1f}%)"
                        else:
                            label_text = f"Unknown ({confidence * 100:.1f}%)"

                        cv2.putText(frame, label_text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Update canvas
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = frame_pil.resize((640, 480))
                    frame_tk = ImageTk.PhotoImage(frame_pil)

                    video_canvas.delete("all")
                    video_canvas.create_image(320, 240, image=frame_tk)
                    video_canvas.image = frame_tk

                # Schedule next update
                if self.realtime_running:
                    self.realtime_window.after(30, update_realtime)

        # Handle window close
        def on_window_close():
            self.stop_realtime_recognition()

        self.realtime_window.protocol("WM_DELETE_WINDOW", on_window_close)

        # Mulai update
        update_realtime()

    def stop_realtime_recognition(self):
        """Stop real-time recognition"""
        self.realtime_running = False
        if hasattr(self, 'realtime_camera') and self.realtime_camera:
            self.realtime_camera.release()
        if hasattr(self, 'realtime_window'):
            self.realtime_window.destroy()

    def tampilkan_eigenfaces(self):
        """Tampilkan eigenfaces (komponen utama)"""
        if not self.model.model_trained:
            messagebox.showerror("Error", "Model belum dilatih")
            return

        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), facecolor='white')

        # Plot eigenfaces
        n_show = min(15, self.model.n_components)
        rows = 3
        cols = 5

        for i in range(n_show):
            ax = fig.add_subplot(rows, cols, i + 1)
            eigenface = self.model.components[i].reshape(self.model.image_shape)
            im = ax.imshow(eigenface, cmap='viridis')
            ax.set_title(f'Komponen #{i + 1}', fontsize=8)
            ax.axis('off')

        fig.suptitle('Eigenfaces (Komponen Utama SVD)', fontsize=14, fontweight='bold')
        fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def evaluasi_model(self):
        """Evaluasi model dengan visualisasi"""
        if not self.model.model_trained:
            messagebox.showerror("Error", "Model belum dilatih")
            return

        if len(self.dataset['images']) < 2:
            messagebox.showerror("Error", "Dataset terlalu kecil untuk evaluasi")
            return

        try:
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                self.dataset['images'],
                self.dataset['labels'],
                test_size=0.3,
                random_state=42,
                stratify=self.dataset['labels'] if len(np.unique(self.dataset['labels'])) > 1 else None
            )

            # Evaluasi
            eval_results = self.model.evaluasi_model(X_test, y_test, self.preprocessing_var.get())

            if eval_results is None:
                messagebox.showerror("Error", "Gagal melakukan evaluasi")
                return

            # Clear previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Create matplotlib figure
            fig = Figure(figsize=(15, 10), facecolor='white')

            # Confusion Matrix
            ax1 = fig.add_subplot(2, 2, 1)
            cm = eval_results['confusion_matrix']

            # Create labels for confusion matrix
            unique_labels = np.unique(np.concatenate([y_test, eval_results['predictions']]))
            labels = [self.dataset['names'].get(label, f'Unknown_{label}') for label in unique_labels]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax1)
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Prediksi')
            ax1.set_ylabel('Aktual')

            # Confidence distribution
            ax2 = fig.add_subplot(2, 2, 2)
            confidences = [c for c in eval_results['confidences'] if c is not None]
            ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Distribusi Confidence')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Frekuensi')
            ax2.axvline(x=np.mean(confidences), color='red', linestyle='--',
                        label=f'Mean: {np.mean(confidences):.3f}')
            ax2.legend()

            # Accuracy per person
            ax3 = fig.add_subplot(2, 2, 3)
            person_accuracies = {}
            for true_label, pred_label in zip(y_test, eval_results['predictions']):
                person_name = self.dataset['names'].get(true_label, f'Unknown_{true_label}')
                if person_name not in person_accuracies:
                    person_accuracies[person_name] = {'correct': 0, 'total': 0}
                person_accuracies[person_name]['total'] += 1
                if true_label == pred_label:
                    person_accuracies[person_name]['correct'] += 1

            names = list(person_accuracies.keys())
            accuracies = [person_accuracies[name]['correct'] / person_accuracies[name]['total']
                          for name in names]

            bars = ax3.bar(names, accuracies, color='lightgreen', alpha=0.7)
            ax3.set_title('Akurasi per Orang')
            ax3.set_ylabel('Akurasi')
            ax3.set_ylim(0, 1)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{acc:.2f}', ha='center', va='bottom')

            # Summary statistics
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')

            summary_text = f"""
            === RINGKASAN EVALUASI ===

            Akurasi Keseluruhan: {eval_results['accuracy']:.4f} ({eval_results['accuracy'] * 100:.2f}%)
            Confidence Rata-rata: {eval_results['avg_confidence']:.4f}

            Total Data Test: {len(y_test)}
            Jumlah Identitas: {len(self.dataset['names'])}

            Preprocessing: {self.preprocessing_var.get()}
            Komponen SVD: {self.model.n_components}
            Threshold: {self.threshold_var.get()}

            === PERFORMA PER KELAS ===
            """

            for name, stats in person_accuracies.items():
                acc = stats['correct'] / stats['total']
                summary_text += f"\n{name}: {acc:.3f} ({stats['correct']}/{stats['total']})"

            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')

            fig.suptitle('Evaluasi Model Pengenalan Wajah SVD', fontsize=16, fontweight='bold')
            fig.tight_layout()

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan evaluasi: {str(e)}")

    def analisis_komponen(self):
        """Analisis pengaruh jumlah komponen SVD"""
        if not self.dataset['images']:
            messagebox.showerror("Error", "Dataset kosong")
            return

        if len(self.dataset['images']) < 4:
            messagebox.showerror("Error", "Dataset terlalu kecil untuk analisis")
            return

        # Progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Analisis Komponen")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)

        ttk.Label(progress_window, text="Sedang menganalisis...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', length=250)
        progress_bar.pack(pady=10)

        def run_analysis():
            try:
                # Range komponen untuk ditest
                component_range = range(5, min(51, len(self.dataset['images'])), 5)
                accuracies = []

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    self.dataset['images'],
                    self.dataset['labels'],
                    test_size=0.3,
                    random_state=42,
                    stratify=self.dataset['labels'] if len(np.unique(self.dataset['labels'])) > 1 else None
                )

                for i, n_comp in enumerate(component_range):
                    # Update progress
                    progress = (i + 1) / len(component_range) * 100
                    progress_bar['value'] = progress
                    progress_window.update()

                    # Test model dengan n_comp komponen
                    test_model = PengenalWajahSVDLanjutan(n_components=n_comp)
                    success, _ = test_model.latih_model(X_train, y_train, self.dataset['names'])

                    if success:
                        eval_results = test_model.evaluasi_model(X_test, y_test)
                        if eval_results:
                            accuracies.append(eval_results['accuracy'])
                        else:
                            accuracies.append(0)
                    else:
                        accuracies.append(0)

                progress_window.destroy()

                # Plot hasil
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()

                fig = Figure(figsize=(12, 8), facecolor='white')

                # Plot accuracy vs components
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.plot(component_range, accuracies, 'bo-', linewidth=2, markersize=8)
                ax1.set_xlabel('Jumlah Komponen SVD')
                ax1.set_ylabel('Akurasi')
                ax1.set_title('Pengaruh Jumlah Komponen SVD terhadap Akurasi')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1)

                # Highlight best performance
                best_idx = np.argmax(accuracies)
                best_comp = list(component_range)[best_idx]
                best_acc = accuracies[best_idx]
                ax1.plot(best_comp, best_acc, 'ro', markersize=12,
                         label=f'Terbaik: {best_comp} komponen ({best_acc:.3f})')
                ax1.legend()

                # Recommendations
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.axis('off')

                recommendations = f"""
                === REKOMENDASI KOMPONEN SVD ===

                Komponen Terbaik: {best_comp}
                Akurasi Terbaik: {best_acc:.4f} ({best_acc * 100:.2f}%)

                === ANALISIS ===
                • Komponen terlalu sedikit: Mungkin kehilangan informasi penting
                • Komponen terlalu banyak: Mungkin overfitting dan noise
                • Optimal: Keseimbangan antara performa dan kompleksitas

                === SARAN ===
                """

                if best_comp <= 15:
                    recommendations += "• Dataset mungkin sederhana, komponen sedikit sudah cukup\n"
                elif best_comp >= 40:
                    recommendations += "• Dataset kompleks, butuh lebih banyak komponen\n"
                else:
                    recommendations += "• Pengaturan komponen dalam range optimal\n"

                recommendations += f"• Gunakan {best_comp} komponen untuk performa terbaik\n"
                recommendations += f"• Monitor overfitting jika dataset kecil\n"

                ax2.text(0.1, 0.9, recommendations, transform=ax2.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace')

                fig.tight_layout()

                # Embed in tkinter
                canvas = FigureCanvasTkAgg(fig, self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Error", f"Gagal melakukan analisis: {str(e)}")

        # Run analysis in thread
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def on_closing(self):
        """Handle aplikasi ditutup"""
        if self.camera_running:
            self.camera_running = False
            if self.camera:
                self.camera.release()

        if hasattr(self, 'realtime_running') and self.realtime_running:
            self.stop_realtime_recognition()

        self.root.destroy()


def main():
    """Fungsi utama untuk menjalankan aplikasi"""
    root = tk.Tk()
    app = AplikasiPengenalWajahGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Set icon jika ada
    try:
        # Anda bisa menambahkan icon file .ico di sini
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()