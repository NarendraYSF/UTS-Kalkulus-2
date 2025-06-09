import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import io
import sys
from contextlib import redirect_stdout
from typing import List, Dict, Optional
import threading

# Import your existing classes (assuming they're in the same file or properly imported)
from svdbiasa import PengenalanWajahSVD, UtilitasMatriks


class SVDFaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Pengenalan Wajah SVD")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the face recognition system
        self.sistem_wajah = PengenalanWajahSVD(ambang_batas=15.0)
        self.is_trained = False
        
        # Create main interface
        self.create_widgets()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Sistem Pengenalan Wajah SVD", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel for controls
        self.create_control_panel(main_frame)
        
        # Right panel for output
        self.create_output_panel(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Siap - Silakan muat database wajah")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Kontrol", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Database section
        db_frame = ttk.LabelFrame(control_frame, text="Database Wajah", padding="5")
        db_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Matrix input
        ttk.Label(db_frame, text="Matriks Database (JSON format):").pack(anchor=tk.W)
        self.matrix_text = scrolledtext.ScrolledText(db_frame, height=6, width=40)
        self.matrix_text.pack(fill=tk.X, pady=5)
        
        # Default example
        default_matrix = '''[[100, 102, 98],
 [120, 118, 122],
 [130, 132, 128],
 [150, 148, 152]]'''
        self.matrix_text.insert(tk.END, default_matrix)
        
        # Labels input
        ttk.Label(db_frame, text="Label (pisahkan dengan koma):").pack(anchor=tk.W, pady=(10, 0))
        self.labels_entry = ttk.Entry(db_frame, width=40)
        self.labels_entry.pack(fill=tk.X, pady=5)
        self.labels_entry.insert(0, "Alice, Bob, Charlie")
        
        # Threshold setting
        threshold_frame = ttk.Frame(db_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        ttk.Label(threshold_frame, text="Ambang Batas:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=15.0)
        threshold_spinbox = ttk.Spinbox(threshold_frame, from_=1.0, to=50.0, 
                                       textvariable=self.threshold_var, width=10)
        threshold_spinbox.pack(side=tk.RIGHT)
        
        # Database buttons
        btn_frame = ttk.Frame(db_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Muat Database", 
                  command=self.load_database).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Hitung SVD", 
                  command=self.compute_svd).pack(side=tk.LEFT, padx=5)
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X)
        
        ttk.Button(file_btn_frame, text="Simpan Model", 
                  command=self.save_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_btn_frame, text="Muat Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Face recognition section
        recognition_frame = ttk.LabelFrame(control_frame, text="Pengenalan Wajah", padding="5")
        recognition_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(recognition_frame, text="Vektor Wajah Uji:").pack(anchor=tk.W)
        self.test_face_text = scrolledtext.ScrolledText(recognition_frame, height=3, width=40)
        self.test_face_text.pack(fill=tk.X, pady=5)
        self.test_face_text.insert(tk.END, "[110, 130, 125, 135]")
        
        # Recognition buttons
        rec_btn_frame = ttk.Frame(recognition_frame)
        rec_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rec_btn_frame, text="Kenali Wajah", 
                  command=self.recognize_face).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(rec_btn_frame, text="Demo Lengkap", 
                  command=self.run_full_demo).pack(side=tk.LEFT, padx=5)
        
        # Batch testing section
        batch_frame = ttk.LabelFrame(control_frame, text="Pengujian Batch", padding="5")
        batch_frame.pack(fill=tk.X)
        
        ttk.Label(batch_frame, text="Batch Wajah (JSON array):").pack(anchor=tk.W)
        self.batch_text = scrolledtext.ScrolledText(batch_frame, height=4, width=40)
        self.batch_text.pack(fill=tk.X, pady=5)
        
        default_batch = '''[[101, 119, 131, 149],
 [104, 124, 134, 154],
 [200, 300, 400, 500]]'''
        self.batch_text.insert(tk.END, default_batch)
        
        ttk.Button(batch_frame, text="Uji Batch", 
                  command=self.batch_test).pack(pady=5)
    
    def create_output_panel(self, parent):
        """Create the right output panel"""
        output_frame = ttk.LabelFrame(parent, text="Output & Hasil", padding="10")
        output_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(output_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Console output tab
        console_frame = ttk.Frame(notebook)
        notebook.add(console_frame, text="Console Output")
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, 
                                                     font=('Courier', 9))
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Hasil Pengenalan")
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     font=('Arial', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear button
        clear_btn = ttk.Button(output_frame, text="Bersihkan Output", 
                              command=self.clear_output)
        clear_btn.pack(pady=(5, 0))
    
    def capture_output(self, func, *args, **kwargs):
        """Capture function output and display in console"""
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            result = func(*args, **kwargs)
            output = captured_output.getvalue()
            
            # Display in console
            self.console_text.insert(tk.END, output + "\n")
            self.console_text.see(tk.END)
            
            return result
        except Exception as e:
            error_msg = f"Error: {str(e)}\n"
            self.console_text.insert(tk.END, error_msg)
            self.console_text.see(tk.END)
            messagebox.showerror("Error", str(e))
            return None
        finally:
            sys.stdout = old_stdout
    
    def load_database(self):
        """Load face database from input"""
        try:
            # Parse matrix
            matrix_str = self.matrix_text.get(1.0, tk.END).strip()
            matrix = eval(matrix_str)  # Note: In production, use json.loads for safety
            
            # Parse labels
            labels_str = self.labels_entry.get().strip()
            labels = [label.strip() for label in labels_str.split(',')] if labels_str else None
            
            # Update threshold
            self.sistem_wajah.ambang_batas = self.threshold_var.get()
            
            # Load database
            self.sistem_wajah.muat_database_dari_matriks(matrix, labels)
            
            self.status_var.set(f"Database dimuat: {len(matrix[0])} wajah, {len(matrix)} dimensi")
            self.console_text.insert(tk.END, f"✓ Database berhasil dimuat\n")
            self.console_text.insert(tk.END, f"  - Jumlah wajah: {len(matrix[0])}\n")
            self.console_text.insert(tk.END, f"  - Dimensi: {len(matrix)}\n")
            self.console_text.insert(tk.END, f"  - Label: {labels}\n\n")
            self.console_text.see(tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat database: {str(e)}")
            self.status_var.set("Error memuat database")
    
    def compute_svd(self):
        """Compute SVD decomposition"""
        if not self.sistem_wajah.database_wajah:
            messagebox.showwarning("Peringatan", "Muat database terlebih dahulu")
            return
        
        self.status_var.set("Menghitung SVD...")
        self.root.update()
        
        # Run in thread to prevent GUI freezing
        def compute_thread():
            try:
                self.capture_output(self.sistem_wajah.hitung_svd)
                self.is_trained = True
                self.status_var.set("SVD berhasil dihitung - Sistem siap untuk pengenalan")
                
                # Update results tab
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "✓ Model SVD berhasil dilatih\n")
                self.results_text.insert(tk.END, f"✓ Ambang batas: {self.sistem_wajah.ambang_batas}\n")
                self.results_text.insert(tk.END, f"✓ Database: {len(self.sistem_wajah.database_wajah)} wajah\n\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menghitung SVD: {str(e)}")
                self.status_var.set("Error menghitung SVD")
        
        threading.Thread(target=compute_thread, daemon=True).start()
    
    def recognize_face(self):
        """Recognize a single face"""
        if not self.is_trained:
            messagebox.showwarning("Peringatan", "Hitung SVD terlebih dahulu")
            return
        
        try:
            # Parse test face vector
            face_str = self.test_face_text.get(1.0, tk.END).strip()
            face_vector = eval(face_str)  # Note: In production, use json.loads for safety
            
            self.status_var.set("Mengenali wajah...")
            self.root.update()
            
            # Perform recognition
            result = self.capture_output(self.sistem_wajah.kenali_wajah, face_vector, True)
            
            if result:
                # Update results tab
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "=== HASIL PENGENALAN WAJAH ===\n\n")
                self.results_text.insert(tk.END, f"Input: {face_vector}\n\n")
                self.results_text.insert(tk.END, f"Status: {'✓ DIKENALI' if result['dikenali'] else '✗ TIDAK DIKENALI'}\n")
                self.results_text.insert(tk.END, f"Error Rekonstruksi: {result['error_rekonstruksi']:.4f}\n")
                self.results_text.insert(tk.END, f"Ambang Batas: {result['ambang_batas']:.4f}\n")
                self.results_text.insert(tk.END, f"Tingkat Keyakinan: {result['tingkat_keyakinan']:.2%}\n\n")
                
                if result['kecocokan_terdekat']['indeks'] >= 0:
                    self.results_text.insert(tk.END, f"Kecocokan Terdekat: {result['kecocokan_terdekat']['label']}\n")
                    self.results_text.insert(tk.END, f"Jarak Kecocokan: {result['kecocokan_terdekat']['jarak']:.4f}\n")
                
                self.status_var.set("Pengenalan selesai")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal mengenali wajah: {str(e)}")
            self.status_var.set("Error pengenalan wajah")
    
    def batch_test(self):
        """Perform batch face recognition"""
        if not self.is_trained:
            messagebox.showwarning("Peringatan", "Hitung SVD terlebih dahulu")
            return
        
        try:
            # Parse batch faces
            batch_str = self.batch_text.get(1.0, tk.END).strip()
            batch_faces = eval(batch_str)  # Note: In production, use json.loads for safety
            
            self.status_var.set("Menjalankan pengujian batch...")
            self.root.update()
            
            # Clear results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=== HASIL PENGUJIAN BATCH ===\n\n")
            
            # Process each face
            for i, face in enumerate(batch_faces):
                self.console_text.insert(tk.END, f"\n--- Menguji Wajah {i+1}: {face} ---\n")
                result = self.sistem_wajah.kenali_wajah(face, verbose=False)
                
                status = "✓ DIKENALI" if result['dikenali'] else "✗ TIDAK DIKENALI"
                self.results_text.insert(tk.END, f"Wajah {i+1}: {face}\n")
                self.results_text.insert(tk.END, f"  Status: {status}\n")
                self.results_text.insert(tk.END, f"  Error: {result['error_rekonstruksi']:.4f}\n")
                self.results_text.insert(tk.END, f"  Keyakinan: {result['tingkat_keyakinan']:.1%}\n")
                
                if result['kecocokan_terdekat']['indeks'] >= 0:
                    self.results_text.insert(tk.END, f"  Terdekat: {result['kecocokan_terdekat']['label']}\n")
                
                self.results_text.insert(tk.END, "\n")
            
            self.console_text.see(tk.END)
            self.status_var.set("Pengujian batch selesai")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menjalankan pengujian batch: {str(e)}")
            self.status_var.set("Error pengujian batch")
    
    def run_full_demo(self):
        """Run the complete demo"""
        self.status_var.set("Menjalankan demo lengkap...")
        self.root.update()
        
        def demo_thread():
            try:
                # Import and run demo functions
                from svdbiasa import demo_pengenalan_wajah, demo_pengenalan_batch
                
                self.console_text.insert(tk.END, "\n=== MENJALANKAN DEMO LENGKAP ===\n\n")
                self.capture_output(demo_pengenalan_wajah)
                self.capture_output(demo_pengenalan_batch)
                
                self.status_var.set("Demo lengkap selesai")
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menjalankan demo: {str(e)}")
                self.status_var.set("Error menjalankan demo")
        
        threading.Thread(target=demo_thread, daemon=True).start()
    
    def save_model(self):
        """Save trained model"""
        if not self.is_trained:
            messagebox.showwarning("Peringatan", "Tidak ada model untuk disimpan")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.sistem_wajah.simpan_model(filename)
                messagebox.showinfo("Sukses", f"Model disimpan ke {filename}")
                self.status_var.set(f"Model disimpan: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan model: {str(e)}")
    
    def load_model(self):
        """Load trained model"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.sistem_wajah.muat_model(filename)
                self.is_trained = True
                
                # Update threshold display
                self.threshold_var.set(self.sistem_wajah.ambang_batas)
                
                messagebox.showinfo("Sukses", f"Model dimuat dari {filename}")
                self.status_var.set(f"Model dimuat: {filename}")
                
                # Update console
                self.console_text.insert(tk.END, f"✓ Model dimuat dari {filename}\n")
                self.console_text.insert(tk.END, f"✓ Sistem siap untuk pengenalan\n\n")
                self.console_text.see(tk.END)
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")
    
    def clear_output(self):
        """Clear all output areas"""
        self.console_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Output dibersihkan")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = SVDFaceRecognitionGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()