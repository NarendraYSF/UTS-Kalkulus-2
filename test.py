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
                         args=(model_name, use_optimized, params)).start()

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

    def _run_evaluation_thread(self, model_name, cv_folds, test_size, seed, k, metrics):
        """Background thread for model evaluation"""
        try:
            # Create algorithm instance
            if model_name == "SVD":
                algo = SVD()
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
        item_ids = np.random.choice