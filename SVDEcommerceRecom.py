import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD, SVDpp, SlopeOne, NMF, CoClustering
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.model_selection import train_test_split
from surprise import accuracy
import warnings
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinter.font import Font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # for Windows 8.1 or later
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # for Windows 7
    except Exception:
        pass

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')


class SVDRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("E-Commerce SVD Recommendation System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f0f0f0")
        self.root.minsize(1000, 650)

        # Define styles and colors
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#e8eef1"
        self.accent_color = "#5cb85c"
        self.text_color = "#333333"

        # Initialize variables
        self.df = None
        self.data = None
        self.model = None
        self.results = {}
        self.best_params = {}
        self.trainset = None
        self.testset = None

        # Setup styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background=self.secondary_color)
        self.style.configure('TLabel', background=self.secondary_color, foreground=self.text_color)
        self.style.configure('TButton', background=self.primary_color, foreground='white')
        self.style.configure('Accent.TButton', background=self.accent_color)
        self.style.map('TButton',
                       background=[('active', '#3d5c8c')],
                       foreground=[('active', 'white')])

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

        # Set default status
        self.update_status("Ready. Start by generating or loading data in the Data tab.")

    def create_header(self):
        """Create application header with title and description"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create title
        title_font = Font(family="Helvetica", size=16, weight="bold")
        title = ttk.Label(header_frame, text="E-Commerce Recommendation System using SVD", font=title_font)
        title.pack(pady=(5, 2))

        # Create subtitle
        subtitle = ttk.Label(header_frame, text="Based on research by Wervyan Shalannanda et al.")
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

        # Add tabs to notebook
        self.notebook.add(self.data_tab, text="  Data  ")
        self.notebook.add(self.algorithm_tab, text="  Algorithm Comparison  ")
        self.notebook.add(self.optimization_tab, text="  Optimization  ")
        self.notebook.add(self.recommendation_tab, text="  Recommendations  ")

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
        # Left panel (controls)
        left_frame = ttk.Frame(self.data_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)

        # Section: Data Generation
        ttk.Label(left_frame, text="Data Generation", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky=tk.W,
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

        # Generate button
        generate_btn = ttk.Button(left_frame, text="Generate Data", command=self.generate_data)
        generate_btn.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(10, 20))

        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Data Statistics Section
        ttk.Label(left_frame, text="Data Statistics", font=("Helvetica", 12, "bold")).grid(row=6, column=0, sticky=tk.W,
                                                                                           pady=(0, 5))

        # Statistics Text
        self.stats_text = scrolledtext.ScrolledText(left_frame, width=30, height=10, wrap=tk.WORD)
        self.stats_text.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW, pady=(0, 10))
        self.stats_text.config(state=tk.DISABLED)

        # Right panel (visualization)
        right_frame = ttk.Frame(self.data_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10, expand=True)

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

        ttk.Label(top_frame, text="Compare Different Matrix Factorization Algorithms",
                  font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, pady=(0, 5))

        # Compare button
        self.compare_btn = ttk.Button(top_frame, text="Run Comparison", command=self.compare_algorithms)
        self.compare_btn.pack(side=tk.RIGHT, padx=(10, 0))

        # Results panel
        results_frame = ttk.Frame(self.algorithm_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left side: Results table
        table_frame = ttk.Frame(results_frame)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        ttk.Label(table_frame, text="Algorithm Performance Metrics", font=("Helvetica", 11, "bold")).pack(anchor=tk.W,
                                                                                                          pady=(0, 5))

        # Create treeview for results
        self.results_tree = ttk.Treeview(table_frame, columns=('rmse', 'fit_time', 'test_time'), show='headings')
        self.results_tree.pack(fill=tk.BOTH, expand=True)

        # Configure columns
        self.results_tree.heading('rmse', text='RMSE')
        self.results_tree.heading('fit_time', text='Fit Time (s)')
        self.results_tree.heading('test_time', text='Test Time (s)')

        self.results_tree.column('rmse', width=100)
        self.results_tree.column('fit_time', width=100)
        self.results_tree.column('test_time', width=100)

        # Right side: Chart area
        chart_frame = ttk.Frame(results_frame)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.algorithm_fig_frame = ttk.Frame(chart_frame)
        self.algorithm_fig_frame.pack(fill=tk.BOTH, expand=True)

        # Initial message
        ttk.Label(self.algorithm_fig_frame, text="Run comparison to view results chart", font=("Helvetica", 11)).pack(
            expand=True)

    def setup_optimization_tab(self):
        """Setup the optimization tab"""
        # Left panel (controls)
        left_frame = ttk.Frame(self.optimization_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)

        # Section: First Optimization
        ttk.Label(left_frame, text="First Optimization Round", font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                                                                    columnspan=2,
                                                                                                    sticky=tk.W,
                                                                                                    pady=(0, 5))

        # Epochs range
        ttk.Label(left_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="5, 10, 15").grid(row=1, column=1, sticky=tk.W, pady=2)

        # Learning rate range
        ttk.Label(left_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="0.002, 0.005, 0.01").grid(row=2, column=1, sticky=tk.W, pady=2)

        # Regularization range
        ttk.Label(left_frame, text="Regularization:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="0.02, 0.4, 0.6").grid(row=3, column=1, sticky=tk.W, pady=2)

        # Optimization button
        opt1_btn = ttk.Button(left_frame, text="Run First Optimization", command=self.run_first_optimization)
        opt1_btn.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(10, 20))

        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # Section: Second Optimization
        ttk.Label(left_frame, text="Second Optimization Round", font=("Helvetica", 12, "bold")).grid(row=6, column=0,
                                                                                                     columnspan=2,
                                                                                                     sticky=tk.W,
                                                                                                     pady=(0, 5))

        # Factors range
        ttk.Label(left_frame, text="Factors:").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="50, 100, 150").grid(row=7, column=1, sticky=tk.W, pady=2)

        # Second round button
        opt2_btn = ttk.Button(left_frame, text="Run Second Optimization", command=self.run_second_optimization)
        opt2_btn.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=(10, 20))

        # Optimization results area
        ttk.Label(left_frame, text="Optimization Results", font=("Helvetica", 12, "bold")).grid(row=9, column=0,
                                                                                                columnspan=2,
                                                                                                sticky=tk.W,
                                                                                                pady=(0, 5))

        self.opt_results_text = scrolledtext.ScrolledText(left_frame, width=30, height=10, wrap=tk.WORD)
        self.opt_results_text.grid(row=10, column=0, columnspan=2, sticky=tk.NSEW)
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

        get_rec_btn = ttk.Button(user_select_frame, text="Get Recommendations", command=self.get_recommendations)
        get_rec_btn.pack(side=tk.RIGHT)

        # Recommendations list
        ttk.Label(user_frame, text="Top Recommendations:").pack(anchor=tk.W, padx=5, pady=(10, 5))

        self.rec_tree = ttk.Treeview(user_frame, columns=('item', 'rating'), show='headings')
        self.rec_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.rec_tree.heading('item', text='Item ID')
        self.rec_tree.heading('rating', text='Predicted Rating')

        self.rec_tree.column('item', width=150)
        self.rec_tree.column('rating', width=150)

        # Right panel: Prediction accuracy
        pred_frame = ttk.LabelFrame(content_frame, text="Prediction Analysis")
        pred_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        # Tabs for best/worst predictions
        pred_notebook = ttk.Notebook(pred_frame)
        pred_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Best predictions tab
        best_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(best_frame, text="Best Predictions")

        self.best_tree = ttk.Treeview(best_frame, columns=('user', 'item', 'actual', 'pred', 'error'), show='headings')
        self.best_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

        # Worst predictions tab
        worst_frame = ttk.Frame(pred_notebook)
        pred_notebook.add(worst_frame, text="Worst Predictions")

        self.worst_tree = ttk.Treeview(worst_frame, columns=('user', 'item', 'actual', 'pred', 'error'),
                                       show='headings')
        self.worst_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

    def generate_synthetic_ecommerce_data(self, num_users=500, num_items=100, sparsity=0.9, seed=42):
        """Generate synthetic e-commerce dataset"""
        np.random.seed(seed)

        # Calculate how many ratings we'll generate
        num_ratings = int(num_users * num_items * (1 - sparsity))

        # Generate random users and items
        user_ids = np.random.choice(num_users, num_ratings)
        item_ids = np.random.choice(num_items, num_ratings)

        # Generate ratings with a skewed distribution (mostly high ratings, as in the paper)
        # The paper mentions: 54.6% are 5-star ratings, 24.6% are 4-star, etc.
        probabilities = [0.035, 0.059, 0.114, 0.246, 0.546]  # From the pie chart in the paper
        ratings = np.random.choice([1, 2, 3, 4, 5], num_ratings, p=probabilities)

        # Create DataFrame
        df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in user_ids],
            'item_id': [f'item_{i}' for i in item_ids],
            'rating': ratings
        })

        # Remove duplicates (same user rating same item multiple times)
        df = df.drop_duplicates(['user_id', 'item_id'])

        return df

    def generate_data(self):
        """Generate synthetic data based on user parameters"""
        self.update_status("Generating synthetic data...", start_progress=True)

        # Get parameters from UI
        num_users = self.num_users_var.get()
        num_items = self.num_items_var.get()
        sparsity = self.sparsity_var.get()

        try:
            # Run data generation in a separate thread
            threading.Thread(target=self._generate_data_thread,
                             args=(num_users, num_items, sparsity)).start()
        except Exception as e:
            self.update_status(f"Error generating data: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")

    def _generate_data_thread(self, num_users, num_items, sparsity):
        """Background thread for data generation"""
        try:
            # Generate data
            self.df = self.generate_synthetic_ecommerce_data(num_users, num_items, sparsity)

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

        # Format statistics text
        stats = f"Total ratings: {total_ratings}\n"
        stats += f"Unique users: {unique_users}\n"
        stats += f"Unique items: {unique_items}\n\n"
        stats += "Rating Distribution:\n"
        for rating, percentage in rating_dist.items():
            stats += f"Rating {rating}: {percentage:.2f}%\n"

        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)

        # Update visualization
        self._update_data_visualization()

        # Update user dropdown in recommendation tab
        # Get all unique users and increase the limit to 100 (or remove limit by removing the slice)
        all_users = self.df['user_id'].unique().tolist()
        # Sort the users for easier navigation
        all_users.sort()
        # Limit to first 100 for balance between choice and performance
        self.user_combo['values'] = all_users[:100]
        if len(self.user_combo['values']) > 0:
            self.user_combo.current(0)

        self.update_status(f"Data generated: {total_ratings} ratings from {unique_users} users on {unique_items} items")

    def _update_data_visualization(self):
        """Update the data visualization chart"""
        # Clear previous chart
        for widget in self.data_fig_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Plot rating distribution
        sns.countplot(x='rating', data=self.df, palette='viridis', ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')

        # Embed in UI
        canvas = FigureCanvasTkAgg(fig, master=self.data_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def compare_algorithms(self):
        """Compare different matrix factorization algorithms"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please generate data first.")
            return

        self.update_status("Comparing algorithms... This may take a while.", start_progress=True)

        # Disable comparison button
        self.compare_btn.config(state=tk.DISABLED)

        # Run comparison in a separate thread
        threading.Thread(target=self._compare_algorithms_thread).start()

    def _compare_algorithms_thread(self):
        """Background thread for algorithm comparison"""
        try:
            # Define algorithms to compare
            algorithms = {
                'SVD': SVD(),
                'SVDpp': SVDpp(),
                'NMF': NMF(),
                'SlopeOne': SlopeOne(),
                'CoClustering': CoClustering()
            }

            # Run cross-validation for each algorithm
            self.results = {}
            for name, algorithm in algorithms.items():
                # Update status
                self.root.after(0, lambda msg=f"Evaluating {name}...": self.update_status(msg, True))

                # Run cross-validation
                cv_results = cross_validate(algorithm, self.data, measures=['RMSE'], cv=5, verbose=False)

                # Store results - handle both array-like and tuple results
                # Calculate mean manually if needed
                self.results[name] = {}

                # Handle test_rmse
                if hasattr(cv_results['test_rmse'], 'mean'):
                    self.results[name]['test_rmse'] = cv_results['test_rmse'].mean()
                else:
                    # If it's a tuple or other iterable without mean method
                    self.results[name]['test_rmse'] = sum(cv_results['test_rmse']) / len(cv_results['test_rmse'])

                # Handle fit_time
                if hasattr(cv_results['fit_time'], 'mean'):
                    self.results[name]['fit_time'] = cv_results['fit_time'].mean()
                else:
                    self.results[name]['fit_time'] = sum(cv_results['fit_time']) / len(cv_results['fit_time'])

                # Handle test_time
                if hasattr(cv_results['test_time'], 'mean'):
                    self.results[name]['test_time'] = cv_results['test_time'].mean()
                else:
                    self.results[name]['test_time'] = sum(cv_results['test_time']) / len(cv_results['test_time'])

            # Update UI in main thread
            self.root.after(0, self._update_algorithm_ui)
        except Exception as e:
            error_msg = str(e)  # Capture the error message
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Failed to compare algorithms: {err}"))
            self.root.after(0, lambda: self.compare_btn.config(state=tk.NORMAL))

    def _update_algorithm_ui(self):
        """Update UI with algorithm comparison results"""
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Insert results into tree
        for name, result in self.results.items():
            self.results_tree.insert('', tk.END, text=name, values=(
                f"{result['test_rmse']:.6f}",
                f"{result['fit_time']:.6f}",
                f"{result['test_time']:.6f}"
            ), iid=name)

        # Update visualization
        self._update_algorithm_visualization()

        # Re-enable comparison button
        self.compare_btn.config(state=tk.NORMAL)

        self.update_status("Algorithm comparison completed.")

    def _update_algorithm_visualization(self):
        """Update the algorithm comparison visualization"""
        # Clear previous chart
        for widget in self.algorithm_fig_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Plot RMSE comparison
        names = list(self.results.keys())
        rmse_values = [self.results[name]['test_rmse'] for name in names]

        sns.barplot(x=names, y=rmse_values, palette='viridis', ax=ax)
        ax.set_title('RMSE Comparison Between Algorithms')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('RMSE (lower is better)')

        # Add value labels
        for i, v in enumerate(rmse_values):
            ax.text(i, v + 0.02, f"{v:.4f}", ha='center')

        plt.tight_layout()

        # Embed in UI
        canvas = FigureCanvasTkAgg(fig, master=self.algorithm_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_first_optimization(self):
        """Run first round of SVD hyperparameter optimization"""
        if self.data is None:
            messagebox.showwarning("Warning", "Please generate data first.")
            return

        self.update_status("Running first optimization round... This may take a while.", start_progress=True)

        # Run optimization in a separate thread
        threading.Thread(target=self._run_first_optimization_thread).start()

    def _run_first_optimization_thread(self):
        """Background thread for first optimization round"""
        try:
            # Define parameter grid
            param_grid = {
                'n_epochs': [5, 10, 15],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.4, 0.6]
            }

            # Run grid search
            gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
            gs.fit(self.data)

            # Store best parameters - make sure we're accessing properly
            try:
                # First try the expected API
                if isinstance(gs.best_params, dict) and 'rmse' in gs.best_params:
                    self.best_params['first_round'] = gs.best_params['rmse']
                else:
                    self.best_params['first_round'] = gs.best_params

                if isinstance(gs.best_score, dict) and 'rmse' in gs.best_score:
                    self.best_rmse_first = gs.best_score['rmse']
                else:
                    self.best_rmse_first = gs.best_score
            except (TypeError, KeyError, AttributeError) as e:
                # If that fails, try alternative access pattern
                # Sometimes GridSearchCV returns results in different format
                if hasattr(gs, 'best_params'):
                    if isinstance(gs.best_params, dict):
                        if 'rmse' in gs.best_params:
                            self.best_params['first_round'] = gs.best_params['rmse']
                        else:
                            self.best_params['first_round'] = gs.best_params
                    else:
                        raise ValueError(f"Unexpected format for best_params: {type(gs.best_params)}")
                else:
                    raise ValueError("gs.best_params is not available")

                if hasattr(gs, 'best_score'):
                    if isinstance(gs.best_score, (int, float)):
                        self.best_rmse_first = gs.best_score
                    elif isinstance(gs.best_score, dict) and 'rmse' in gs.best_score:
                        self.best_rmse_first = gs.best_score['rmse']
                    else:
                        raise ValueError(f"Unexpected format for best_score: {type(gs.best_score)}")
                else:
                    raise ValueError("gs.best_score is not available")

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_ui('first'))
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Failed to run optimization: {err}"))

    def run_second_optimization(self):
        """Run second round of SVD hyperparameter optimization"""
        if not hasattr(self, 'best_params') or 'first_round' not in self.best_params:
            messagebox.showwarning("Warning", "Please run first optimization round first.")
            return

        self.update_status("Running second optimization round... This may take a while.", start_progress=True)

        # Run optimization in a separate thread
        threading.Thread(target=self._run_second_optimization_thread).start()

    def _run_second_optimization_thread(self):
        """Background thread for second optimization round"""
        try:
            # Get best parameters from first round
            first_round_params = self.best_params['first_round']

            # Check if parameters exist and get them safely
            best_epochs = first_round_params.get('n_epochs', 10)  # Default to 10 if missing
            best_lr = first_round_params.get('lr_all', 0.005)  # Default to 0.005 if missing
            best_reg = first_round_params.get('reg_all', 0.4)  # Default to 0.4 if missing

            # Define parameter grid
            param_grid = {
                'n_factors': [50, 100, 150],
                'n_epochs': [max(5, best_epochs - 5), best_epochs, best_epochs + 5],
                'lr_all': [best_lr * 0.5, best_lr, best_lr * 1.5],
                'reg_all': [max(0.01, best_reg - 0.2), best_reg, min(1.0, best_reg + 0.2)]
            }

            # Run grid search
            gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
            gs.fit(self.data)

            # Store best parameters - make sure we're accessing properly
            try:
                # First try the expected API
                if isinstance(gs.best_params, dict) and 'rmse' in gs.best_params:
                    self.best_params['second_round'] = gs.best_params['rmse']
                else:
                    self.best_params['second_round'] = gs.best_params

                if isinstance(gs.best_score, dict) and 'rmse' in gs.best_score:
                    self.best_rmse_second = gs.best_score['rmse']
                else:
                    self.best_rmse_second = gs.best_score
            except (TypeError, KeyError, AttributeError) as e:
                # If that fails, try alternative access pattern
                if hasattr(gs, 'best_params'):
                    if isinstance(gs.best_params, dict):
                        if 'rmse' in gs.best_params:
                            self.best_params['second_round'] = gs.best_params['rmse']
                        else:
                            self.best_params['second_round'] = gs.best_params
                    else:
                        raise ValueError(f"Unexpected format for best_params: {type(gs.best_params)}")
                else:
                    raise ValueError("gs.best_params is not available")

                if hasattr(gs, 'best_score'):
                    if isinstance(gs.best_score, (int, float)):
                        self.best_rmse_second = gs.best_score
                    elif isinstance(gs.best_score, dict) and 'rmse' in gs.best_score:
                        self.best_rmse_second = gs.best_score['rmse']
                    else:
                        raise ValueError(f"Unexpected format for best_score: {type(gs.best_score)}")
                else:
                    raise ValueError("gs.best_score is not available")

            # Update UI in main thread
            self.root.after(0, lambda: self._update_optimization_ui('second'))
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0,
                            lambda err=error_msg: messagebox.showerror("Error", f"Failed to run optimization: {err}"))

    def _update_optimization_ui(self, round_name):
        """Update UI with optimization results"""
        # Update results text
        self.opt_results_text.config(state=tk.NORMAL)

        if round_name == 'first':
            # Clear previous text
            self.opt_results_text.delete(1.0, tk.END)

            # Add first round results
            self.opt_results_text.insert(tk.END, "First Optimization Round:\n")
            self.opt_results_text.insert(tk.END, f"Best RMSE: {self.best_rmse_first:.6f}\n\n")
            self.opt_results_text.insert(tk.END, "Best Parameters:\n")

            for param, value in self.best_params['first_round'].items():
                self.opt_results_text.insert(tk.END, f"{param}: {value}\n")

        elif round_name == 'second':
            # Add second round results
            self.opt_results_text.insert(tk.END, "\nSecond Optimization Round:\n")
            self.opt_results_text.insert(tk.END, f"Best RMSE: {self.best_rmse_second:.6f}\n\n")
            self.opt_results_text.insert(tk.END, "Best Parameters:\n")

            for param, value in self.best_params['second_round'].items():
                self.opt_results_text.insert(tk.END, f"{param}: {value}\n")

            # Update visualization
            self._update_optimization_visualization()

        self.opt_results_text.config(state=tk.DISABLED)

        self.update_status(f"{round_name.capitalize()} optimization round completed.")

    def _update_optimization_visualization(self):
        """Update the optimization visualization"""
        # Clear previous chart
        for widget in self.opt_fig_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        # Get RMSE values
        if 'SVD' in self.results:
            base_svd_rmse = self.results['SVD']['test_rmse']

            # Prepare data for visualization
            names = ['SVD (Base)', 'SVD (First Opt)', 'SVD (Final Opt)']
            rmse_values = [base_svd_rmse, self.best_rmse_first, self.best_rmse_second]

            # Plot RMSE comparison
            bars = sns.barplot(x=names, y=rmse_values, palette='viridis', ax=ax)
            ax.set_title('RMSE Improvement with Optimization')
            ax.set_xlabel('Model')
            ax.set_ylabel('RMSE (lower is better)')

            # Add value labels
            for i, v in enumerate(rmse_values):
                ax.text(i, v + 0.01, f"{v:.6f}", ha='center')

            # Add percentage improvement
            improvement = ((base_svd_rmse - self.best_rmse_second) / base_svd_rmse) * 100
            ax.annotate(f"Total Improvement: {improvement:.2f}%",
                        xy=(0.5, 0.05), xycoords='figure fraction',
                        ha='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

            plt.tight_layout()

            # Embed in UI
            canvas = FigureCanvasTkAgg(fig, master=self.opt_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def train_final_model(self):
        """Train the final optimized model"""
        if not hasattr(self, 'best_params') or 'second_round' not in self.best_params:
            messagebox.showwarning("Warning", "Please complete optimization first.")
            return

        self.update_status("Training final model...", start_progress=True)

        # Run training in a separate thread
        threading.Thread(target=self._train_final_model_thread).start()

    def _train_final_model_thread(self):
        """Background thread for training final model"""
        try:
            # Create model with best parameters - safely getting parameters with defaults if missing
            second_round_params = self.best_params['second_round']

            self.model = SVD(
                n_factors=second_round_params.get('n_factors', 100),
                n_epochs=second_round_params.get('n_epochs', 20),
                lr_all=second_round_params.get('lr_all', 0.005),
                reg_all=second_round_params.get('reg_all', 0.4)
            )

            # Split data into train and test sets
            self.trainset, self.testset = train_test_split(self.data, test_size=0.25)

            # Train the model
            self.model.fit(self.trainset)

            # Make predictions on test set
            predictions = self.model.test(self.testset)

            # Check if predictions is in the expected format
            if not hasattr(predictions, '__iter__') or isinstance(predictions, (int, float)):
                raise ValueError(f"Predictions not in expected format: {type(predictions)}")

            # Calculate RMSE
            self.final_rmse = accuracy.rmse(predictions)

            # Store predictions for displaying best/worst
            self.predictions = predictions

            # Update UI in main thread
            self.root.after(0, self._update_final_model_ui)
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error", f"Failed to train model: {err}"))

    def _update_final_model_ui(self):
        """Update UI after final model training"""
        # Update status
        self.update_status(f"Final model trained. RMSE: {self.final_rmse:.6f}")

        # Update prediction analysis trees
        self._update_prediction_analysis()

        # Enable recommendation button
        self.train_btn.config(state=tk.NORMAL)

    def _update_prediction_analysis(self):
        """Update prediction analysis trees with best and worst predictions"""
        # Clear previous entries
        for tree in [self.best_tree, self.worst_tree]:
            for item in tree.get_children():
                tree.delete(item)

        # Check if predictions is a valid object to sort
        if not hasattr(self.predictions, '__iter__') or isinstance(self.predictions, (int, float)):
            self.update_status("Error: Predictions not in expected format.")
            messagebox.showerror("Error", "Predictions not in expected format for analysis.")
            return

        # Sort predictions by error
        self.sorted_predictions = sorted(self.predictions, key=lambda x: abs(x.r_ui - x.est))

        # Insert best predictions
        for i, pred in enumerate(self.sorted_predictions[:5]):
            error = abs(pred.r_ui - pred.est)
            self.best_tree.insert('', tk.END, text=f"#{i + 1}", values=(
                pred.uid, pred.iid, f"{pred.r_ui:.1f}", f"{pred.est:.4f}", f"{error:.6f}"
            ))

        # Insert worst predictions
        for i, pred in enumerate(self.sorted_predictions[-5:]):
            error = abs(pred.r_ui - pred.est)
            self.worst_tree.insert('', tk.END, text=f"#{i + 1}", values=(
                pred.uid, pred.iid, f"{pred.r_ui:.1f}", f"{pred.est:.4f}", f"{error:.6f}"
            ))

    def get_recommendations(self):
        """Get recommendations for selected user"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the final model first.")
            return

        user_id = self.user_var.get()
        if not user_id:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        self.update_status(f"Generating recommendations for {user_id}...", start_progress=True)

        # Run in a separate thread
        threading.Thread(target=self._get_recommendations_thread, args=(user_id,)).start()

    def _get_recommendations_thread(self, user_id):
        """Background thread for getting recommendations - simplified approach"""
        try:
            # Get the user's rated items from the dataframe directly
            user_rated_items = set(self.df[self.df['user_id'] == user_id]['item_id'].values)

            # Get all items
            all_items = set(self.df['item_id'].unique())

            # Get unrated items
            unrated_items = all_items - user_rated_items

            # Limit to 100 items if there are too many
            candidate_items = list(unrated_items)[:100]

            if not candidate_items:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No unrated items for this user."))
                self.root.after(0, lambda: self.update_status("Ready"))
                return

            # Generate recommendations using a simpler approach
            recommendations = []

            # Now generate predictions one by one
            for item_id in candidate_items:
                try:
                    # Get prediction
                    prediction = self.model.predict(uid=user_id, iid=item_id)

                    # Extract just the item_id and estimated rating as a tuple
                    item_prediction = (item_id, prediction.est if hasattr(prediction, 'est') else prediction)

                    recommendations.append(item_prediction)
                except Exception as item_err:
                    # Skip problematic items
                    continue

            # Check if we got any recommendations
            if not recommendations:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "Could not generate recommendations."))
                self.root.after(0, lambda: self.update_status("Ready"))
                return

            # Sort by predicted rating (second element in each tuple)
            recommendations.sort(key=lambda x: x[1], reverse=True)

            # Take top 5
            self.recommendations = recommendations[:5]

            # Update UI in main thread
            self.root.after(0, self._update_recommendations_ui)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda err=error_msg: self.update_status(f"Error: {err}"))
            self.root.after(0, lambda err=error_msg: messagebox.showerror("Error",
                                                                          f"Failed to get recommendations: {err}"))

    def _update_recommendations_ui(self):
        """Update UI with recommendations"""
        # Clear previous recommendations
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)

        # Insert recommendations
        for i, (item, rating) in enumerate(self.recommendations):
            self.rec_tree.insert('', tk.END, text=f"#{i + 1}", values=(item, f"{rating:.4f}"))

        self.update_status(f"Generated {len(self.recommendations)} recommendations for {self.user_var.get()}")


def main():
    root = tk.Tk()
    app = SVDRecommenderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()