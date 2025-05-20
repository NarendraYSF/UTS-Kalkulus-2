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

        # Configure default styles
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.primary_color, foreground='white')
        self.style.configure('Accent.TButton', background=self.accent_color)
        self.style.configure('Success.TButton', background=self.success_color)
        self.style.configure('Warning.TButton', background=self.warning_color)
        self.style.configure('Error.TButton', background=self.error_color)

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

        # Update notebook styles
        self.style.configure('TNotebook', background=self.secondary_color)
        self.style.configure('TNotebook.Tab', background=self.primary_color, foreground=self.text_color)

        # Update treeview styles
        self.style.configure('Treeview',
                             background=self.secondary_color,
                             foreground=self.text_color,
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
        file_menu.add_command(label="Export Recommendations", command=self.export_recommendations)
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

    def export_recommendations(self):
        """Export recommendations to a file"""
        if not hasattr(self, 'recommendations') or not self.recommendations:
            messagebox.showwarning("Warning", "No recommendations to export.")
            return

        user_id = self.user_var.get()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"recommendations_{user_id}_{timestamp}.csv"

        file_path = filedialog.asksaveasfilename(
            title="Export Recommendations",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Create a dataframe from recommendations
                rec_df = pd.DataFrame(self.recommendations, columns=['item_id', 'predicted_rating'])
                rec_df['user_id'] = user_id
                rec_df['timestamp'] = datetime.datetime.now()
                rec_df['model'] = self.current_model_name

                # Save to CSV
                rec_df.to_csv(file_path, index=False)

                self.update_status(f"Recommendations exported to {file_path}")
                messagebox.showinfo("Export Successful", f"Recommendations exported to {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export recommendations: {str(e)}")

    def save_model(self):
        """Save the current model to a file"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model to save. Please train a model first.")
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
                # Save model, parameters, and metadata
                model_data = {
                    'model': self.model,
                    'model_name': self.current_model_name,
                    'parameters': self.best_params.get('second_round', {}),
                    'metrics': self.evaluation_metrics.get(self.current_model_name, {}),
                    'timestamp': datetime.datetime.now().isoformat(),
                    'data_shape': (len(self.df), self.df['user_id'].nunique(), self.df['item_id'].nunique())
                }

                with open(file_path, 'wb') as f:
                    pickle.dump(model_data, f)

                self.last_saved_path = file_path
                self.update_status(f"Model saved to {file_path}")
                messagebox.showinfo("Save Successful", f"Model saved to {file_path}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        """Load a model from a file"""
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.update_status("Loading model...", start_progress=True)

                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Extract model and metadata
                self.model = model_data['model']
                self.current_model_name = model_data['model_name']

                if 'parameters' in model_data:
                    if 'second_round' not in self.best_params:
                        self.best_params['second_round'] = {}
                    self.best_params['second_round'] = model_data['parameters']

                if 'metrics' in model_data and self.current_model_name in model_data['metrics']:
                    if self.current_model_name not in self.evaluation_metrics:
                        self.evaluation_metrics[self.current_model_name] = {}
                    self.evaluation_metrics[self.current_model_name] = model_data['metrics']

                # Update status
                model_info = f"Model: {self.current_model_name}"
                if 'data_shape' in model_data:
                    total, users, items = model_data['data_shape']
                    model_info += f", trained on {total} ratings from {users} users and {items} items"

                if 'timestamp' in model_data:
                    try:
                        dt = datetime.datetime.fromisoformat(model_data['timestamp'])
                        model_info += f", saved on {dt.strftime('%Y-%m-%d %H:%M:%S')}"
                    except:
                        pass

                self.update_status(f"Model loaded: {model_info}")

                # Update UI if needed
                if hasattr(self, 'model_info_label'):
                    self.model_info_label.config(text=model_info)

                messagebox.showinfo("Load Successful", f"Model loaded: {self.current_model_name}")

            except Exception as e:
                self.update_status("Ready")
                messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")

    def on_exit(self):
        """Handle application exit"""
        # Save any unsaved data or configuration
        self.save_config()

        # Stop any running threads
        for thread in self.active_threads:
            if thread.is_alive():
                # Cannot forcibly terminate threads in Python
                # We rely on thread's cooperative termination
                pass

        # Close the application
        self.root.destroy()

    def show_documentation(self):
        """Show documentation in web browser"""
        # Documentation URL or local file
        doc_url = "https://surprise.readthedocs.io/en/stable/"

        try:
            webbrowser.open(doc_url)
        except:
            messagebox.showerror("Error", "Could not open documentation.")

    def show_about(self):
        """Show about dialog"""
        about_dialog = tk.Toplevel(self.root)
        about_dialog.title("About Enhanced SVD Recommender")
        about_dialog.geometry("500x400")
        about_dialog.transient(self.root)
        about_dialog.grab_set()

        # Header
        ttk.Label(about_dialog, text="Enhanced SVD Recommender",
                  font=("Helvetica", 16, "bold")).pack(pady=(20, 10))

        ttk.Label(about_dialog, text="Version 2.0").pack(pady=(0, 20))

        # Description
        description = (
            "This application implements various recommendation algorithms based on "
            "matrix factorization techniques, particularly Singular Value Decomposition (SVD). "
            "It provides tools for data exploration, algorithm comparison, hyperparameter "
            "optimization, and recommendation generation for e-commerce applications."
        )

        desc_label = ttk.Label(about_dialog, text=description, wraplength=400, justify="center")
        desc_label.pack(pady=(0, 20), padx=20)

        # Credits
        ttk.Label(about_dialog, text="Based on research by Wervyan Shalannanda et al.",
                  font=("Helvetica", 10, "italic")).pack(pady=(0, 5))

        ttk.Label(about_dialog, text="Powered by Surprise library",
                  font=("Helvetica", 10)).pack(pady=(0, 20))

        # Close button
        ttk.Button(about_dialog, text="Close", command=about_dialog.destroy).pack(pady=(10, 20))

    def create_header(self):
        """Create application header with title and description"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create title
        title_font = Font(family="Helvetica", size=16, weight="bold")
        title = ttk.Label(header_frame, text="Enhanced E-Commerce Recommendation System", font=title_font)
        title.pack(pady=(5, 2))

        # Create subtitle
        subtitle = ttk.Label(header_frame, text="Featuring SVD, SVD++, NMF, KNN and more...")
        subtitle.pack(pady=(0, 5))

    def create_notebook(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.algorithm_tab = ttk.Frame(self.notebook)
        self.optimization_tab = ttk.Frame(self.notebook)
        self.recommendation_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        # Add tabs to notebook
        self.notebook.add(self.data_tab, text="  Data  ")
        self.notebook.add(self.algorithm_tab, text="  Algorithm Comparison  ")
        self.notebook.add(self.optimization_tab, text="  Optimization  ")
        self.notebook.add(self.recommendation_tab, text="  Recommendations  ")
        self.notebook.add(self.evaluation_tab, text="  Evaluation  ")
        self.notebook.add(self.visualization_tab, text="  Visualization  ")
        self.notebook.add(self.settings_tab, text="  Settings  ")

    def create_status_bar(self):
        """Create status bar at bottom of window"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate', length=150)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))

    def update_status(self, message, start_progress=False):
        """Update status bar message and progress indicator"""
        self.status_var.set(message)
        self.root.update_idletasks()

        if start_progress:
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()

    def setup_data_tab(self):
        """Setup the data tab with controls and visualization"""
        # Top frame for buttons
        top_frame = ttk.Frame(self.data_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(top_frame, text="Generate Random Data", command=self.generate_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top_frame, text="Import Data", command=self.import_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top_frame, text="Export Data", command=self.export_data).pack(side=tk.LEFT)

        # Content frame
        content_frame = ttk.Frame(self.data_tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel (controls)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=10, expand=False)

        # Section: Data Generation
        ttk.Label(left_frame, text="Data Generation Parameters", font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                                                                      columnspan=2,
                                                                                                      sticky=tk.W,
                                                                                                      pady=(0, 5))

        # Number of users
        ttk.Label(left_frame, text="Number of Users:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.num_users_var = tk.IntVar(value=500)
        ttk.Entry(left_frame, textvariable=self.num_users_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        # Number of items
        ttk.Label(left_frame, text="Number of Items:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.num_items_var = tk.IntVar(value=100)
        ttk.Entry(left_frame, textvariable=self.num_items_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Sparsity
        ttk.Label(left_frame, text="Sparsity (0-1):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.sparsity_var = tk.DoubleVar(value=0.9)
        ttk.Entry(left_frame, textvariable=self.sparsity_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)

        # Rating distribution
        ttk.Label(left_frame, text="Rating Distribution:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.rating_dist_var = tk.StringVar(value="skewed")
        rating_combo = ttk.Combobox(left_frame, textvariable=self.rating_dist_var, width=10, state="readonly")
        rating_combo['values'] = ("skewed", "normal", "uniform")
        rating_combo.grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(left_frame, text="Seed:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(left_frame, textvariable=self.seed_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)

        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Data Statistics Section
        ttk.Label(left_frame, text="Data Statistics", font=("Helvetica", 12, "bold")).grid(row=7, column=0,
                                                                                           columnspan=2, sticky=tk.W,
                                                                                           pady=(0, 5))

        # Statistics Text
        self.stats_text = scrolledtext.ScrolledText(left_frame, width=30, height=15, wrap=tk.WORD)
        self.stats_text.grid(row=8, column=0, columnspan=2, sticky=tk.NSEW, pady=(0, 10))
        self.stats_text.config(state=tk.DISABLED)

        # Right panel (visualization)
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=0, pady=10, expand=True)

        # Chart area
        self.data_fig_frame = ttk.Frame(right_frame)
        self.data_fig_frame.pack(fill=tk.BOTH, expand=True)

        # Initial message
        ttk.Label(self.data_fig_frame, text="Generate data to view distribution chart", font=("Helvetica", 11)).pack(
            expand=True)

    def setup_algorithm_tab(self):
        """Setup the algorithm comparison tab"""
        # Top controls panel
        top_frame = ttk.Frame(self.algorithm_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Compare Different Recommendation Algorithms", font=("Helvetica", 12, "bold")).pack(
            side=tk.LEFT, pady=(0, 5))

        # Compare button
        self.compare_btn = ttk.Button(top_frame, text="Run Comparison", command=self.compare_algorithms)
        self.compare_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Algorithm selection
        alg_frame = ttk.Frame(self.algorithm_tab)
        alg_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(alg_frame, text="Select Algorithms to Compare:").pack(side=tk.LEFT, padx=(0, 10))

        # Checkboxes for algorithms
        self.alg_vars = {}
        algs = [
            ("SVD", "SVD"),
            ("SVD++", "SVDpp"),
            ("NMF", "NMF"),
            ("SlopeOne", "SlopeOne"),
            ("CoClustering", "CoClustering"),
            ("KNN Basic", "KNNBasic"),
            ("KNN with Means", "KNNWithMeans"),
            ("KNN with Z-Score", "KNNWithZScore")
        ]

        for i, (name, class_name) in enumerate(algs):
            var = tk.BooleanVar(value=True if i < 4 else False)  # First 4 selected by default
            self.alg_vars[class_name] = var

            cb = ttk.Checkbutton(alg_frame, text=name, variable=var)
            cb.pack(side=tk.LEFT, padx=5)

        # Results panel
        results_frame = ttk.Frame(self.algorithm_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side: Results table
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        ttk.Label(table_frame, text="Algorithm Performance Metrics", font=("Helvetica", 11, "bold")).pack(anchor=tk.W,
                                                                                                          pady=(0, 5))

        # Create treeview for results
        self.results_tree = ttk.Treeview(table_frame, columns=('rmse', 'mae', 'fit_time', 'test_time'), show='headings')
        self.results_tree.pack(fill=tk.BOTH, expand=True)

        # Configure columns
        self.results_tree.heading('rmse', text='RMSE')
        self.results_tree.heading('mae', text='MAE')
        self.results_tree.heading('fit_time', text='Fit Time (s)')
        self.results_tree.heading('test_time', text='Test Time (s)')

        self.results_tree.column('rmse', width=100)
        self.results_tree.column('mae', width=100)
        self.results_tree.column('fit_time', width=100)
        self.results_tree.column('test_time', width=100)

        # Add vertical scrollbar
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)

        # Right side: Chart tabs
        chart_frame = ttk.Frame(results_frame)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a notebook for different chart views
        chart_notebook = ttk.Notebook(chart_frame)
        chart_notebook.pack(fill=tk.BOTH, expand=True)

        # RMSE Chart tab
        self.rmse_chart_frame = ttk.Frame(chart_notebook)
        chart_notebook.add(self.rmse_chart_frame, text="RMSE Comparison")

        # Time Chart tab
        self.time_chart_frame = ttk.Frame(chart_notebook)
        chart_notebook.add(self.time_chart_frame, text="Time Comparison")

        # Metrics tab
        self.metrics_chart_frame = ttk.Frame(chart_notebook)
        chart_notebook.add(self.metrics_chart_frame, text="Metrics")

        # Initial message
        self.algorithm_fig_frame = ttk.Frame(self.rmse_chart_frame)
        self.algorithm_fig_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.algorithm_fig_frame, text="Run comparison to view results chart", font=("Helvetica", 11)).pack(
            expand=True)

    def setup_optimization_tab(self):
        """Setup the optimization tab"""
        # Top frame for buttons
        top_frame = ttk.Frame(self.optimization_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Model Selection and Hyperparameter Optimization",
                  font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)

        # Left panel (controls)
        left_frame = ttk.Frame(self.optimization_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)

        # Model selection
        ttk.Label(left_frame, text="Select Model to Optimize:", font=("Helvetica", 11)).grid(row=0, column=0,
                                                                                             columnspan=2, sticky=tk.W,
                                                                                             pady=(0, 5))

        self.opt_model_var = tk.StringVar(value="SVD")
        model_combo = ttk.Combobox(left_frame, textvariable=self.opt_model_var, width=15, state="readonly")
        model_combo['values'] = ("SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans")
        model_combo.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Section: First Optimization
        ttk.Label(left_frame, text="First Optimization Round", font=("Helvetica", 11, "bold")).grid(row=2, column=0,
                                                                                                    columnspan=2,
                                                                                                    sticky=tk.W,
                                                                                                    pady=(0, 5))

        # Epochs range
        ttk.Label(left_frame, text="Epochs:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="5, 10, 15")
        ttk.Entry(left_frame, textvariable=self.epochs_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=2)

        # Learning rate range
        ttk.Label(left_frame, text="Learning Rate:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.StringVar(value="0.002, 0.005, 0.01")
        ttk.Entry(left_frame, textvariable=self.lr_var, width=15).grid(row=4, column=1, sticky=tk.W, pady=2)

        # Regularization range
        ttk.Label(left_frame, text="Regularization:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.reg_var = tk.StringVar(value="0.02, 0.4, 0.6")
        ttk.Entry(left_frame, textvariable=self.reg_var, width=15).grid(row=5, column=1, sticky=tk.W, pady=2)

        # Optimization button
        opt1_btn = ttk.Button(left_frame, text="Run First Optimization", command=self.run_first_optimization)
        opt1_btn.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(10, 20))

        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Section: Second Optimization
        ttk.Label(left_frame, text="Second Optimization Round", font=("Helvetica", 11, "bold")).grid(row=8, column=0,
                                                                                                     columnspan=2,
                                                                                                     sticky=tk.W,
                                                                                                     pady=(0, 5))

        # Factors range
        ttk.Label(left_frame, text="Factors:").grid(row=9, column=0, sticky=tk.W, pady=2)
        self.factors_var = tk.StringVar(value="50, 100, 150")
        ttk.Entry(left_frame, textvariable=self.factors_var, width=15).grid(row=9, column=1, sticky=tk.W, pady=2)

        # Second round button
        opt2_btn = ttk.Button(left_frame, text="Run Second Optimization", command=self.run_second_optimization)
        opt2_btn.grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=(10, 20))

        # Optimization results area
        ttk.Label(left_frame, text="Optimization Results", font=("Helvetica", 11, "bold")).grid(row=11, column=0,
                                                                                                columnspan=2,
                                                                                                sticky=tk.W,
                                                                                                pady=(0, 5))

        self.opt_results_text = scrolledtext.ScrolledText(left_frame, width=30, height=10, wrap=tk.WORD)
        self.opt_results_text.grid(row=12, column=0, columnspan=2, sticky=tk.NSEW)
        self.opt_results_text.config(state=tk.DISABLED)

        # Right panel (visualization)
        right_frame = ttk.Frame(self.optimization_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10, expand=True)

        # Comparison chart
        self.opt_fig_frame = ttk.Frame(right_frame)
        self.opt_fig_frame.pack(fill=tk.BOTH, expand=True)

        # Initial message
        ttk.Label(self.opt_fig_frame, text="Run optimization to view comparison chart", font=("Helvetica", 11)).pack(
            expand=True)

    def setup_recommendation_tab(self):
        """Setup the recommendation tab"""
        # Top control panel
        top_frame = ttk.Frame(self.recommendation_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Generate Recommendations", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT,
                                                                                                   pady=(0, 5))

        # Train model button
        self.train_btn = ttk.Button(top_frame, text="Train Final Model", command=self.train_final_model)
        self.train_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Model selection
        model_frame = ttk.Frame(top_frame)
        model_frame.pack(side=tk.RIGHT, padx=(0, 10))

        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))

        self.model_var = tk.StringVar(value="SVD")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=12, state="readonly")
        model_combo['values'] = ("SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans")
        model_combo.pack(side=tk.LEFT)

        # Model info display
        self.model_info_label = ttk.Label(self.recommendation_tab, text="Train model to see details")
        self.model_info_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Main content
        content_frame = ttk.Frame(self.recommendation_tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: User recommendation
        user_frame = ttk.LabelFrame(content_frame, text="Recommendations for User")
        user_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        # User selection
        user_select_frame = ttk.Frame(user_frame)
        user_select_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(user_select_frame, text="User:").pack(side=tk.LEFT, padx=(0, 5))
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(user_select_frame, textvariable=self.user_var, state="readonly")
        self.user_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        # Add search functionality
        ttk.Label(user_select_frame, text="or Search:").pack(side=tk.LEFT, padx=(5, 5))
        self.user_search_var = tk.StringVar()
        self.user_search_var.trace_add("write", self.filter_users)
        user_search = ttk.Entry(user_select_frame, textvariable=self.user_search_var, width=15)
        user_search.pack(side=tk.LEFT, padx=(0, 5))

        get_rec_btn = ttk.Button(user_select_frame, text="Get Recommendations", command=self.get_recommendations)
        get_rec_btn.pack(side=tk.RIGHT)

        # Recommendations list
        rec_frame = ttk.Frame(user_frame)
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 5))

        ttk.Label(rec_frame, text="Top Recommendations:").pack(anchor=tk.W, pady=(0, 5))

        rec_tree_frame = ttk.Frame(rec_frame)
        rec_tree_frame.pack(fill=tk.BOTH, expand=True)

        self.rec_tree = ttk.Treeview(rec_tree_frame, columns=('item', 'rating'), show='headings')
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.rec_tree.heading('item', text='Item ID')
        self.rec_tree.heading('rating', text='Predicted Rating')

        self.rec_tree.column('item', width=150)
        self.rec_tree.column('rating', width=150)

        # Add scrollbar
        rec_scrollbar = ttk.Scrollbar(rec_tree_frame, orient="vertical", command=self.rec_tree.yview)
        rec_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rec_tree.configure(yscrollcommand=rec_scrollbar.set)

        # Export button
        ttk.Button(user_frame, text="Export Recommendations", command=self.export_recommendations).pack(
            side=tk.BOTTOM, anchor=tk.E, padx=5, pady=5)

        # Right panel: Tabs for different recommendation types
        rec_tabs_frame = ttk.Frame(content_frame)
        rec_tabs_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        rec_notebook = ttk.Notebook(rec_tabs_frame)
        rec_notebook.pack(fill=tk.BOTH, expand=True)

        # Prediction analysis tab
        pred_frame = ttk.Frame(rec_notebook)
        rec_notebook.add(pred_frame, text="Prediction Analysis")

        # Sub-tabs for best/worst predictions
        pred_notebook = ttk.Notebook(pred_frame)
        pred_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Best predictions tab
        best_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(best_frame, text="Best Predictions")

        best_tree_frame = ttk.Frame(best_frame)
        best_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.best_tree = ttk.Treeview(best_tree_frame, columns=('user', 'item', 'actual', 'pred', 'error'),
                                      show='headings')
        self.best_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.best_tree.heading('user', text='User')
        self.best_tree.heading('item', text='Item')
        self.best_tree.heading('actual', text='Actual')
        self.best_tree.heading('pred', text='Prediction')
        self.best_tree.heading('error', text='Error')

        self.best_tree.column('user', width=100)
        self.best_tree.column('item', width=100)
        self.best_tree.column('actual', width=70)
        self.best_tree.column('pred', width=70)
        self.best_tree.column('error', width=70)

        # Add scrollbar
        best_scrollbar = ttk.Scrollbar(best_tree_frame, orient="vertical", command=self.best_tree.yview)
        best_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.best_tree.configure(yscrollcommand=best_scrollbar.set)

        # Worst predictions tab
        worst_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(worst_frame, text="Worst Predictions")

        worst_tree_frame = ttk.Frame(worst_frame)
        worst_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.worst_tree = ttk.Treeview(worst_tree_frame, columns=('user', 'item', 'actual', 'pred', 'error'),
                                       show='headings')
        self.worst_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.worst_tree.heading('user', text='User')
        self.worst_tree.heading('item', text='Item')
        self.worst_tree.heading('actual', text='Actual')
        self.worst_tree.heading('pred', text='Prediction')
        self.worst_tree.heading('error', text='Error')

        self.worst_tree.column('user', width=100)
        self.worst_tree.column('item', width=100)
        self.worst_tree.column('actual', width=70)
        self.worst_tree.column('pred', width=70)
        self.worst_tree.column('error', width=70)

        # Add scrollbar
        worst_scrollbar = ttk.Scrollbar(worst_tree_frame, orient="vertical", command=self.worst_tree.yview)
        worst_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.worst_tree.configure(yscrollcommand=worst_scrollbar.set)

        # Item-based recommendations tab
        item_frame = ttk.Frame(rec_notebook)
        rec_notebook.add(item_frame, text="Item-Based Recommendations")

        # Item selection
        item_select_frame = ttk.Frame(item_frame)
        item_select_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Label(item_select_frame, text="Item:").pack(side=tk.LEFT, padx=(0, 5))
        self.item_var = tk.StringVar()
        self.item_combo = ttk.Combobox(item_select_frame, textvariable=self.item_var, state="readonly")
        self.item_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(item_select_frame, text="Find Similar Items", command=self.find_similar_items).pack(side=tk.RIGHT)

        # Similar items table
        similar_frame = ttk.Frame(item_frame)
        similar_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(10, 5))

        ttk.Label(similar_frame, text="Items Most Similar To Selected Item:").pack(anchor=tk.W, pady=(0, 5))

        self.similar_tree = ttk.Treeview(similar_frame, columns=('item', 'similarity'), show='headings')
        self.similar_tree.pack(fill=tk.BOTH, expand=True)

        self.similar_tree.heading('item', text='Item ID')
        self.similar_tree.heading('similarity', text='Similarity Score')

        self.similar_tree.column('item', width=150)
        self.similar_tree.column('similarity', width=150)

    def setup_evaluation_tab(self):
        """Setup the evaluation tab with metrics and cross-validation"""
        # Top controls panel
        top_frame = ttk.Frame(self.evaluation_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Model Evaluation and Validation", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT,
                                                                                                          pady=(0, 5))

        # Run evaluation button
        self.eval_btn = ttk.Button(top_frame, text="Run Evaluation", command=self.run_evaluation)
        self.eval_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Content panel
        content_frame = ttk.Frame(self.evaluation_tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: Controls and parameters
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), pady=0)

        # Evaluation settings
        ttk.Label(left_frame, text="Evaluation Settings", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Model selection
        model_frame = ttk.Frame(left_frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.eval_model_var = tk.StringVar(value="SVD")
        model_combo = ttk.Combobox(model_frame, textvariable=self.eval_model_var, width=15, state="readonly")
        model_combo['values'] = ("SVD", "SVDpp", "NMF", "KNNBasic", "KNNWithMeans", "KNNWithZScore")
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        # Cross-validation settings
        ttk.Label(model_frame, text="CV Folds:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.cv_folds_var = tk.IntVar(value=5)
        ttk.Spinbox(model_frame, from_=2, to=10, textvariable=self.cv_folds_var, width=5).grid(row=1, column=1,
                                                                                               sticky=tk.W, padx=(5, 0),
                                                                                               pady=(5, 0))

        # Test size
        ttk.Label(model_frame, text="Test Size:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.test_size_var = tk.DoubleVar(value=0.25)
        test_size_combo = ttk.Combobox(model_frame, textvariable=self.test_size_var, width=5, state="readonly")
        test_size_combo['values'] = (0.1, 0.2, 0.25, 0.3, 0.4, 0.5)
        test_size_combo.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))

        # Random state
        ttk.Label(model_frame, text="Random State:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.eval_seed_var = tk.IntVar(value=42)
        ttk.Entry(model_frame, textvariable=self.eval_seed_var, width=5).grid(row=3, column=1, sticky=tk.W, padx=(5, 0),
                                                                              pady=(5, 0))

        # Metrics to calculate
        ttk.Label(left_frame, text="Metrics", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(15, 5))

        metrics_frame = ttk.Frame(left_frame)
        metrics_frame.pack(fill=tk.X)

        # Checkboxes for metrics
        self.metric_vars = {}
        metrics = [
            ("RMSE", "rmse", True),
            ("MAE", "mae", True),
            ("Precision@k", "precision", True),
            ("Recall@k", "recall", True),
            ("F1@k", "f1", True),
            ("Coverage", "coverage", False)
        ]

        for i, (name, key, default) in enumerate(metrics):
            var = tk.BooleanVar(value=default)
            self.metric_vars[key] = var

            ttk.Checkbutton(metrics_frame, text=name, variable=var).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=5,
                                                                         pady=2)

        # k value for precision/recall
        k_frame = ttk.Frame(left_frame)
        k_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(k_frame, text="k value:").pack(side=tk.LEFT, padx=(0, 5))
        self.k_var = tk.IntVar(value=5)
        ttk.Spinbox(k_frame, from_=1, to=50, textvariable=self.k_var, width=5).pack(side=tk.LEFT)

        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # Results section
        ttk.Label(left_frame, text="Evaluation Results", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.eval_results_text = scrolledtext.ScrolledText(left_frame, width=30, height=10, wrap=tk.WORD)
        self.eval_results_text.pack(fill=tk.BOTH, expand=True)
        self.eval_results_text.config(state=tk.DISABLED)

        # Right panel: Visualizations
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Create a notebook for different visualization tabs
        viz_notebook = ttk.Notebook(right_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)

        # ROC Curve tab
        self.roc_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.roc_frame, text="ROC Curve")

        # Precision-Recall tab
        self.pr_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.pr_frame, text="Precision-Recall")

        # Cross-validation tab
        self.cv_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(self.cv_frame, text="Cross-Validation")

        # Initial message
        ttk.Label(self.roc_frame, text="Run evaluation to view ROC curve", font=("Helvetica", 11)).pack(expand=True)
        ttk.Label(self.pr_frame, text="Run evaluation to view precision-recall curve", font=("Helvetica", 11)).pack(
            expand=True)
        ttk.Label(self.cv_frame, text="Run evaluation to view cross-validation results", font=("Helvetica", 11)).pack(
            expand=True)

    def setup_visualization_tab(self):
        """Setup the visualization tab with advanced charts"""
        # Top controls panel
        top_frame = ttk.Frame(self.visualization_tab)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top_frame, text="Advanced Data Visualization", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT,
                                                                                                      pady=(0, 5))

        # Content panel with chart types
        content_frame = ttk.Frame(self.visualization_tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel: Chart controls
        control_frame = ttk.Frame(content_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(control_frame, text="Chart Type", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Chart type selection
        self.chart_type_var = tk.StringVar(value="distribution")
        charts = [
            ("Rating Distribution", "distribution"),
            ("User Activity", "user_activity"),
            ("Item Popularity", "item_popularity"),
            ("Rating Heatmap", "heatmap"),
            ("User Similarity", "user_similarity"),
            ("Item Similarity", "item_similarity")
        ]

        for text, value in charts:
            ttk.Radiobutton(control_frame, text=text, value=value, variable=self.chart_type_var).pack(anchor=tk.W,
                                                                                                      pady=2)

        ttk.Button(control_frame, text="Generate Visualization", command=self.generate_visualization).pack(pady=15)

        # Chart options section
        ttk.Label(control_frame, text="Chart Options", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(15, 5))

        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X)

        # Sample size option
        ttk.Label(options_frame, text="Sample Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sample_size_var = tk.IntVar(value=1000)
        ttk.Entry(options_frame, textvariable=self.sample_size_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=2)

        # Color scheme
        ttk.Label(options_frame, text="Color Scheme:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.color_scheme_var = tk.StringVar(value="viridis")
        scheme_combo = ttk.Combobox(options_frame, textvariable=self.color_scheme_var, width=8, state="readonly")
        scheme_combo['values'] = ("viridis", "plasma", "inferno", "magma", "cividis", "Set1", "Set2", "Set3", "tab10")
        scheme_combo.grid(row=1, column=1, sticky=tk.W, pady=2)

        # Figure size
        ttk.Label(options_frame, text="Figure Width:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.fig_width_var = tk.IntVar(value=10)
        ttk.Spinbox(options_frame, from_=5, to=20, textvariable=self.fig_width_var, width=5).grid(row=2, column=1,
                                                                                                  sticky=tk.W, pady=2)

        ttk.Label(options_frame, text="Figure Height:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.fig_height_var = tk.IntVar(value=6)
        ttk.Spinbox(options_frame, from_=3, to=15, textvariable=self.fig_height_var, width=5).grid(row=3, column=1,
                                                                                                   sticky=tk.W, pady=2)

        # Export options
        ttk.Label(control_frame, text="Export", font=("Helvetica", 11, "bold")).pack(anchor=tk.W, pady=(15, 5))

        ttk.Button(control_frame, text="Export Chart", command=self.export_chart).pack(anchor=tk.W)

        # Right panel: Chart display
        self.chart_frame = ttk.Frame(content_frame)
        self.chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Initial message
        ttk.Label(self.chart_frame, text="Select chart type and click Generate Visualization",
                  font=("Helvetica", 11)).pack(expand=True)

    def setup_settings_tab(self):
        """Setup the settings tab"""
        # Top header
        ttk.Label(self.settings_tab, text="Application Settings",
                  font=("Helvetica", 14, "bold")).pack(pady=(20, 10))

        # Content frame
        content_frame = ttk.Frame(self.settings_tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Appearance section
        ttk.Label(content_frame, text="Appearance", font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Theme selection
        theme_frame = ttk.Frame(content_frame)
        theme_frame.grid(row=1, column=0, sticky=tk.W, pady=5)

        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=(0, 10))

        self.theme_var = tk.StringVar(value="light" if not self.is_dark_mode else "dark")
        ttk.Radiobutton(theme_frame, text="Light", value="light", variable=self.theme_var,
                        command=lambda: self.apply_light_theme()).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(theme_frame, text="Dark", value="dark", variable=self.theme_var,
                        command=lambda: self.apply_dark_theme()).pack(side=tk.LEFT, padx=5)

        # Custom colors
        color_frame = ttk.Frame(content_frame)
        color_frame.grid(row=2, column=0, sticky=tk.W, pady=5)

        ttk.Label(color_frame, text="Accent Color:").pack(side=tk.LEFT, padx=(0, 10))

        self.accent_color_btn = ttk.Button(color_frame, text="Choose Color",
                                           command=self.choose_accent_color)
        self.accent_color_btn.pack(side=tk.LEFT)

        # Performance section
        ttk.Label(content_frame, text="Performance", font=("Helvetica", 12, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(20, 10))

        # Memory usage
        mem_frame = ttk.Frame(content_frame)
        mem_frame.grid(row=4, column=0, sticky=tk.W, pady=5)

        ttk.Label(mem_frame, text="Memory Management:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(mem_frame, text="Clear Cache", command=self.clear_cache).pack(side=tk.LEFT, padx=(0, 5))

        # Limit threads
        thread_frame = ttk.Frame(content_frame)
        thread_frame.grid(row=5, column=0, sticky=tk.W, pady=5)

        ttk.Label(thread_frame, text="Maximum Worker Threads:").pack(side=tk.LEFT, padx=(0, 10))
        self.max_threads_var = tk.IntVar(value=4)
        ttk.Spinbox(thread_frame, from_=1, to=16, textvariable=self.max_threads_var, width=5).pack(side=tk.LEFT)

        # Data handling section
        ttk.Label(content_frame, text="Data Handling", font=("Helvetica", 12, "bold")).grid(
            row=6, column=0, sticky=tk.W, pady=(20, 10))

        # Default data directory
        dir_frame = ttk.Frame(content_frame)
        dir_frame.grid(row=7, column=0, sticky=tk.W, pady=5)

        ttk.Label(dir_frame, text="Default Data Directory:").pack(side=tk.LEFT, padx=(0, 10))

        self.data_dir_var = tk.StringVar(value=os.path.expanduser("~"))
        ttk.Entry(dir_frame, textvariable=self.data_dir_var, width=30).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self.choose_data_dir).pack(side=tk.LEFT)

        # Recent files
        recent_frame = ttk.Frame(content_frame)
        recent_frame.grid(row=8, column=0, sticky=tk.W, pady=5)

        ttk.Label(recent_frame, text="Recent Files:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(recent_frame, text="Clear Recent Files", command=self.clear_recent_files).pack(side=tk.LEFT)

        # Right side: About and system info
        right_frame = ttk.Frame(content_frame)
        right_frame.grid(row=0, column=1, rowspan=9, padx=(50, 0), sticky=tk.NSEW)

        ttk.Label(right_frame, text="About", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        about_text = (
            "Enhanced SVD Recommender v2.0\n\n"
            "This application implements various recommendation algorithms\n"
            "based on matrix factorization and collaborative filtering.\n\n"
            "Author: Claude\n"
            "Date: May 2025\n\n"
            "Built with: Python, tkinter, Surprise, matplotlib"
        )

        ttk.Label(right_frame, text=about_text, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 20))

        # System info
        ttk.Label(right_frame, text="System Information", font=("Helvetica", 12, "bold")).pack(anchor=tk.W,
                                                                                               pady=(0, 10))

        import platform
        system_info = (
            f"OS: {platform.system()} {platform.release()}\n"
            f"Python: {platform.python_version()}\n"
            f"Processor: {platform.processor()}"
        )

        ttk.Label(right_frame, text=system_info, justify=tk.LEFT).pack(anchor=tk.W)

    def choose_accent_color(self):
        """Open color chooser to select accent color"""
        color = colorchooser.askcolor(initialcolor=self.accent_color)
        if color[1]:
            self.accent_color = color[1]
            # Update styles
            self.style.configure('Accent.TButton', background=self.accent_color)

    def clear_cache(self):
        """Clear application cache"""
        # Reset any cached data
        self.results = {}
        self.best_params = {}

        # Force garbage collection
        import gc
        gc.collect()

        messagebox.showinfo("Cache Cleared", "Application cache has been cleared.")

    def choose_data_dir(self):
        """Choose default data directory"""
        directory = filedialog.askdirectory(initialdir=self.data_dir_var.get())
        if directory:
            self.data_dir_var.set(directory)

    def filter_users(self, *args):
        """Filter users in dropdown based on search text"""
        if not hasattr(self, 'df') or self.df is None:
            return

        search_text = self.user_search_var.get().lower()

        if not search_text:
            # If search is empty, reset to default limited list
            all_users = self.df['user_id'].unique().tolist()
            all_users.sort()
            self.user_combo['values'] = all_users[:100]
            return

        # Filter users based on search text
        filtered_users = [user for user in self.df['user_id'].unique() if search_text in str(user).lower()]
        filtered_users.sort()

        # Limit to first 100 matches
        self.user_combo['values'] = filtered_users[:100]

        # If only one match, select it
        if len(filtered_users) == 1:
            self.user_combo.set(filtered_users[0])

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
        self.update_status("Generating synthetic data...", start_progress=True)

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
                             args=(model_name, epochs, lr_values, reg_values)).start()

        def _run_first_optimization_thread(self, model_name, epochs, lr_values, reg_values):
            """Background thread for first optimization round"""
            try:
                # Define parameter grid
                param_grid = {
                    'n_epochs': epochs,
                    'lr_all': lr_values,
                    'reg_all': reg_values
                }

                # Create model based on selection
                if model_name == "SVD":
                    algo = SVD()
                elif model_name == "SVDpp":
                    algo = SVDpp()
                elif model_name == "NMF":
                    algo = NMF()
                elif model_name == "KNNBasic":
                    algo = KNNBasic()
                elif model_name == "KNNWithMeans":
                    algo = KNNWithMeans()
                else:
                    raise ValueError(f"Model {model_name} not supported for optimization")

                # Run grid search
                gs = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=3)
                gs.fit(self.data)

                # Get best params and scores
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
                    param_str = f"epochs={params['n_epochs']}, lr={params['lr_all']}, reg={params['reg_all']}"
                    param_combos.append(param_str)
                    rmse_means.append(cv_results['mean_rmse'][i])
                    mae_means.append(cv_results['mean_mae'][i])

                # Update UI in main thread
                self.root.after(0, lambda: self._update_optimization_results(
                    'first_round', model_name, best_params, best_rmse, best_mae,
                    param_combos, rmse_means, mae_means))

            except Exception as e:
                error_msg = str(e)
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
        except ValueError:
            messagebox.showerror("Error", "Invalid factors values. Please enter comma-separated numbers.")
            return

        self.update_status(f"Running second optimization round for {model_name}...", start_progress=True)

        # Start optimization in a separate thread
        threading.Thread(target=self._run_second_optimization_thread,
                         args=(model_name, best_params, factors)).start()

    def _run_second_optimization_thread(self, model_name, best_params, factors):
        """Background thread for second optimization round"""
        try:
            # Define parameter grid with factors and best params from first round
            param_grid = {
                'n_factors': factors,
                'n_epochs': [best_params['n_epochs']],
                'lr_all': [best_params['lr_all']],
                'reg_all': [best_params['reg_all']]
            }

            # Create model
            if model_name == "SVD":
                algo = SVD()
            elif model_name == "SVDpp":
                algo = SVDpp()
            elif model_name == "NMF":
                algo = NMF()
            else:
                algo = SVD()  # Fallback

            # Run grid search
            gs = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=3)
            gs.fit(self.data)

            # Get best params and scores
            best_params_second = gs.best_params['rmse']
            best_rmse = gs.best_score['rmse']
            best_mae = gs.best_score['mae']

            # Store results
            self.best_params['second_round'] = best_params_second

            # Create results for visualization
            cv_results = gs.cv_results
            param_combos = []
            rmse_means = []
            mae_means = []

            for i, params in enumerate(cv_results['params']):
                param_str = f"factors={params['n_factors']}"
                param_combos.append(param_str)
                rmse_means.append(cv_results['mean_rmse'][i])
                mae_means.append(cv_results['mean_mae'][i])

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_results(
                'second_round', model_name, best_params_second, best_rmse, best_mae,
                param_combos, rmse_means, mae_means))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Optimization failed: {err}"))

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
            # Create model
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
                    algo = NMF(n_factors=params.get('n_factors', 15),
                               n_epochs=params.get('n_epochs', 50),
                               reg_all=params.get('reg_all', 0.06))
                else:
                    algo = NMF()
            elif model_name == "KNNBasic":
                algo = KNNBasic()
            elif model_name == "KNNWithMeans":
                algo = KNNWithMeans()
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


# Main function to run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedSVDRecommenderApp(root)
    root.mainloop()