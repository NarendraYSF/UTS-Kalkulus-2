import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import io
import base64
from surprise import SVD, SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import pickle
import json
from datetime import datetime
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # for Windows 8.1 or later
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # for Windows 7
    except Exception:
        pass

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e9ecef;
        margin: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# Define utility functions
def generate_hash(password):
    """Generate a hash for the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def save_user_data(user_data):
    """Save user data to a JSON file."""
    with open('users.json', 'w') as f:
        json.dump(user_data, f)


def load_user_data():
    """Load user data from a JSON file."""
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_ratings(username, ratings):
    """Save user ratings."""
    user_data = load_user_data()
    if username in user_data:
        user_data[username]['ratings'] = ratings
        save_user_data(user_data)


def load_ratings(username):
    """Load user ratings."""
    user_data = load_user_data()
    if username in user_data and 'ratings' in user_data[username]:
        return user_data[username]['ratings']
    return []


def save_model(model, filename):
    """Save the trained model using pickle."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    """Load a trained model using pickle."""
    try:
        with open(filename, 'rb') as f:
            return pickle.dump(f)
    except (FileNotFoundError, pickle.PickleError):
        return None


def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for the given object."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# Define the MovieRecommender class
class MovieRecommender:
    """A movie recommendation system using SVD and SVD++."""

    def __init__(self, use_svdpp=False):
        """Initialize the recommender system."""
        self.use_svdpp = use_svdpp
        self.model = None
        self.data = None
        self.trainset = None
        self.testset = None
        self.movie_data = None
        self.ratings_data = None
        self.hyperparams = {}
        self.metrics = {'rmse': None, 'mae': None}
        self.cross_val_results = None
        self.user_movie_matrix = None

    def load_movie_data(self, file):
        """Load movie data from a file."""
        try:
            if isinstance(file, str):
                # Load from path
                self.movie_data = pd.read_csv(file, encoding='latin-1',
                                              names=['Movie_Id', 'Year', 'Name'],
                                              header=None)
            else:
                # Load from uploaded file
                self.movie_data = pd.read_csv(file, encoding='latin-1',
                                              names=['Movie_Id', 'Year', 'Name'],
                                              header=None)

            # Clean the data
            self.movie_data['Year'] = pd.to_numeric(self.movie_data['Year'], errors='coerce')
            self.movie_data['Movie_Id'] = pd.to_numeric(self.movie_data['Movie_Id'], errors='coerce')
            self.movie_data.dropna(subset=['Movie_Id'], inplace=True)
            self.movie_data['Movie_Id'] = self.movie_data['Movie_Id'].astype(int)
            self.movie_data.set_index('Movie_Id', inplace=True)

            return True
        except Exception as e:
            st.error(f"Error loading movie data: {str(e)}")
            return False

    def load_ratings_data(self, files, sample_size=None, progress_bar=None):
        """Load ratings data from files."""
        all_ratings = []

        # Process files
        for i, file in enumerate(files):
            if progress_bar:
                progress_bar.progress((i) / len(files))

            try:
                # Process the file
                if isinstance(file, str):
                    # Local file path
                    with open(file, 'r', encoding='utf-8') as f:
                        self._process_ratings_file(f.read(), all_ratings)
                else:
                    # Uploaded file - could be bytes or string
                    try:
                        # Try to get value as bytes first
                        content = file.getvalue()
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        self._process_ratings_file(content, all_ratings)
                    except AttributeError:
                        # If file is not a BytesIO/StringIO object
                        self._process_ratings_file(file, all_ratings)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                continue

        if progress_bar:
            progress_bar.progress(1.0)

        # Check if we have any ratings
        if not all_ratings:
            st.error("No ratings data was loaded. Please check the input files.")
            self.ratings_data = pd.DataFrame()  # Create empty DataFrame
            return 0

        # Convert to DataFrame
        self.ratings_data = pd.DataFrame(all_ratings)

        # Take a sample if specified
        if sample_size and len(self.ratings_data) > sample_size:
            self.ratings_data = self._stratified_sample(self.ratings_data, sample_size)

        # Create a Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(
            self.ratings_data[['Cust_Id', 'Movie_Id', 'Rating']], reader
        )

        # Split into training and test sets
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2, random_state=42)

        return len(self.ratings_data)

    def _process_ratings_file(self, file_content, all_ratings):
        """Process a ratings file and append ratings to the list."""
        movie_id = None

        # Convert file_content to lines
        if isinstance(file_content, list):
            lines = file_content
        elif isinstance(file_content, bytes):
            lines = file_content.decode('utf-8').splitlines()
        else:
            # Already a string
            lines = file_content.splitlines()

        for line in lines:
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            line = line.strip()
            if line.endswith(':'):
                movie_id = line[:-1]
            else:
                # Parse customer ID, rating, and date
                parts = line.split(',')
                if len(parts) >= 2 and movie_id:
                    try:
                        customer_id, rating = parts[0], parts[1]
                        all_ratings.append({
                            'Movie_Id': int(movie_id),
                            'Cust_Id': int(customer_id),
                            'Rating': float(rating)
                        })
                    except (ValueError, IndexError) as e:
                        # Skip invalid lines silently or log them
                        pass

    def _stratified_sample(self, df, sample_size):
        """Take a stratified sample of the ratings data."""
        # Group by ratings to maintain distribution
        rating_counts = df['Rating'].value_counts(normalize=True)

        # Sample from each rating category
        sampled = pd.DataFrame()
        for rating, proportion in rating_counts.items():
            rating_df = df[df['Rating'] == rating]
            n = int(sample_size * proportion)
            if n > len(rating_df):
                n = len(rating_df)
            sampled = pd.concat([sampled, rating_df.sample(n, random_state=42)])

        return sampled

    def tune_hyperparameters(self, progress_bar=None):
        """Perform grid search to find optimal hyperparameters."""
        if self.data is None:
            st.error("Data not loaded. Please load data first.")
            return None

        # Parameter grid
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.1]
        }

        # Initialize the algorithm
        algo_class = SVDpp if self.use_svdpp else SVD

        # Perform grid search
        gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=3)

        if progress_bar:
            progress_bar.progress(0.1)

        gs.fit(self.data)

        if progress_bar:
            progress_bar.progress(1.0)

        # Get best parameters
        self.hyperparams = gs.best_params['rmse']

        # Store all results for visualization
        all_results = []
        for params, rmse, mae in zip(gs.cv_results['params'],
                                     gs.cv_results['mean_test_rmse'],
                                     gs.cv_results['mean_test_mae']):
            result = params.copy()
            result['rmse'] = rmse
            result['mae'] = mae
            all_results.append(result)

        return pd.DataFrame(all_results)

    def train_model(self, params=None, progress_bar=None):
        """Train the recommendation model."""
        if self.data is None:
            st.error("Data not loaded. Please load data first.")
            return False

        # Use provided params, stored hyperparams, or defaults
        if params is None:
            params = self.hyperparams if self.hyperparams else {}

        # Initialize the algorithm with parameters
        algo_class = SVDpp if self.use_svdpp else SVD
        self.model = algo_class(
            n_factors=params.get('n_factors', 100),
            n_epochs=params.get('n_epochs', 20),
            lr_all=params.get('lr_all', 0.005),
            reg_all=params.get('reg_all', 0.02)
        )

        # Train on the full trainset
        if progress_bar:
            progress_bar.progress(0.5)

        self.model.fit(self.trainset)

        if progress_bar:
            progress_bar.progress(1.0)

        return True

    def cross_validate_model(self, k=5, progress_bar=None):
        """Perform k-fold cross-validation."""
        if self.data is None:
            st.error("Data not loaded. Please load data first.")
            return None

        # Initialize the algorithm
        algo = SVDpp() if self.use_svdpp else SVD()

        # Perform cross-validation
        if progress_bar:
            progress_bar.progress(0.1)

        cv_results = cross_validate(algo, self.data, measures=['rmse', 'mae'],
                                    cv=k, verbose=False, n_jobs=-1)

        if progress_bar:
            progress_bar.progress(1.0)

        # Calculate average metrics
        self.metrics['rmse'] = np.mean(cv_results['test_rmse'])
        self.metrics['mae'] = np.mean(cv_results['test_mae'])

        # Store results for visualization
        self.cross_val_results = cv_results

        return cv_results

    def evaluate_model(self, progress_bar=None):
        """Evaluate the model on the test set."""
        if self.model is None:
            st.error("Model not trained. Please train the model first.")
            return None

        if progress_bar:
            progress_bar.progress(0.5)

        # Test the model
        predictions = self.model.test(self.testset)

        if progress_bar:
            progress_bar.progress(1.0)

        # Calculate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)

        self.metrics['rmse'] = rmse
        self.metrics['mae'] = mae

        return rmse, mae

    def get_recommendations(self, user_id, n=10, min_rating=0, progress_bar=None):
        """Get movie recommendations for a specific user."""
        if self.model is None:
            st.error("Model not trained. Please train the model first.")
            return None

        if self.movie_data is None:
            st.error("Movie data not loaded.")
            return None

        if progress_bar:
            progress_bar.progress(0.2)

        # Get all movies
        all_movie_ids = self.movie_data.index.unique()

        # Get movies already rated by the user
        user_ratings = []
        for u, i, r in self.trainset.all_ratings():
            if self.trainset.to_raw_uid(u) == user_id:
                user_ratings.append((self.trainset.to_raw_iid(i), r))

        if progress_bar:
            progress_bar.progress(0.4)

        # Movies not rated by the user
        rated_movie_ids = [mid for mid, _ in user_ratings]
        unrated_movies = list(set(all_movie_ids) - set(rated_movie_ids))

        # Predict ratings for unrated movies
        predictions = []
        total = len(unrated_movies)

        for i, movie_id in enumerate(unrated_movies):
            if i % 100 == 0 and progress_bar:
                progress_bar.progress(0.4 + 0.5 * (i / total))

            try:
                # Check if user and movie are in the trainset
                inner_user_id = self.trainset.to_inner_uid(user_id)
                inner_movie_id = self.trainset.to_inner_iid(movie_id)

                # Predict rating
                pred = self.model.predict(user_id, movie_id)

                # Filter by minimum rating
                if pred.est >= min_rating:
                    predictions.append((movie_id, pred.est))

            except (ValueError, KeyError):
                # Skip if user or movie not in trainset
                continue

        if progress_bar:
            progress_bar.progress(0.9)

        # Sort by predicted rating in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Get top N recommendations
        top_n = predictions[:n]

        # Create a DataFrame with recommendations
        recommendations = []
        for movie_id, predicted_rating in top_n:
            try:
                movie_info = self.movie_data.loc[movie_id]
                recommendations.append({
                    'Movie_Id': movie_id,
                    'Name': movie_info['Name'],
                    'Year': movie_info['Year'],
                    'Predicted_Rating': round(predicted_rating, 2)
                })
            except (KeyError, ValueError):
                continue

        if progress_bar:
            progress_bar.progress(1.0)

        return pd.DataFrame(recommendations)

    def create_user_movie_matrix(self):
        """Create a user-movie matrix for visualization."""
        if self.ratings_data is None:
            st.error("Ratings data not loaded.")
            return None

        # Create a pivot table
        user_movie_matrix = self.ratings_data.pivot_table(
            index='Cust_Id',
            columns='Movie_Id',
            values='Rating',
            fill_value=0
        )

        self.user_movie_matrix = user_movie_matrix
        return user_movie_matrix

    def add_user_rating(self, user_id, movie_id, rating):
        """Add a new user rating."""
        if self.ratings_data is None:
            self.ratings_data = pd.DataFrame(columns=['Cust_Id', 'Movie_Id', 'Rating'])

        # Check if the rating already exists
        existing = self.ratings_data[
            (self.ratings_data['Cust_Id'] == user_id) &
            (self.ratings_data['Movie_Id'] == movie_id)
            ]

        if len(existing) > 0:
            # Update existing rating
            self.ratings_data.loc[
                (self.ratings_data['Cust_Id'] == user_id) &
                (self.ratings_data['Movie_Id'] == movie_id),
                'Rating'
            ] = rating
        else:
            # Add new rating
            new_rating = pd.DataFrame([{
                'Cust_Id': user_id,
                'Movie_Id': movie_id,
                'Rating': rating
            }])
            self.ratings_data = pd.concat([self.ratings_data, new_rating], ignore_index=True)

        # Update the Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(
            self.ratings_data[['Cust_Id', 'Movie_Id', 'Rating']], reader
        )

        # Update the trainset
        self.trainset = self.data.build_full_trainset()

        return True

    def get_user_ratings(self, user_id):
        """Get all ratings for a specific user."""
        if self.ratings_data is None:
            return pd.DataFrame()

        user_ratings = self.ratings_data[self.ratings_data['Cust_Id'] == user_id]

        if len(user_ratings) == 0:
            return pd.DataFrame()

        # Merge with movie data
        if self.movie_data is not None:
            ratings_with_info = user_ratings.merge(
                self.movie_data.reset_index(),
                on='Movie_Id',
                how='left'
            )
            return ratings_with_info[['Movie_Id', 'Name', 'Year', 'Rating']]

        return user_ratings

    def get_popular_movies(self, n=20):
        """Get the most popular movies based on number of ratings."""
        if self.ratings_data is None:
            return pd.DataFrame()

        # Count ratings per movie
        movie_counts = self.ratings_data['Movie_Id'].value_counts().reset_index()
        movie_counts.columns = ['Movie_Id', 'Count']

        # Calculate average rating per movie
        movie_avg = self.ratings_data.groupby('Movie_Id')['Rating'].mean().reset_index()
        movie_avg.columns = ['Movie_Id', 'Avg_Rating']

        # Merge count and average
        movie_stats = movie_counts.merge(movie_avg, on='Movie_Id')

        # Get top N movies
        top_movies = movie_stats.sort_values('Count', ascending=False).head(n)

        # Add movie info
        if self.movie_data is not None:
            top_movies = top_movies.merge(
                self.movie_data.reset_index(),
                on='Movie_Id',
                how='left'
            )

        return top_movies


# Main Streamlit App
def main():
    # Session state initialization
    if 'recommender' not in st.session_state:
        st.session_state.recommender = MovieRecommender(use_svdpp=False)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'view' not in st.session_state:
        st.session_state.view = "login"

    # Header
    st.markdown("<h1 class='main-header'>Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3074/3074767.png", width=100)
        st.markdown("## Navigation")

        if st.session_state.authenticated:
            st.success(f"Logged in as: {st.session_state.username}")

            if st.button("Log Out"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.user_id = None
                st.session_state.view = "login"
                st.rerun()

            # Navigation options
            nav_options = ["Dashboard", "Data Management", "Model Training",
                           "Recommendations", "My Ratings", "Analytics"]
            selected_nav = st.selectbox("Go to:", nav_options)

            if selected_nav != st.session_state.view:
                st.session_state.view = selected_nav
                st.rerun()

            # Algorithm selection
            algo_type = st.radio(
                "Recommendation Algorithm:",
                ["SVD", "SVD++"],
                index=1 if st.session_state.recommender.use_svdpp else 0
            )

            if (algo_type == "SVD" and st.session_state.recommender.use_svdpp) or \
                    (algo_type == "SVD++" and not st.session_state.recommender.use_svdpp):
                st.session_state.recommender = MovieRecommender(use_svdpp=(algo_type == "SVD++"))
                st.info(f"Switched to {algo_type} algorithm")
        else:
            st.info("Please log in to access the system")

    # Main content based on the current view
    if not st.session_state.authenticated:
        show_login_signup()
    elif st.session_state.view == "Dashboard":
        show_dashboard()
    elif st.session_state.view == "Data Management":
        show_data_management()
    elif st.session_state.view == "Model Training":
        show_model_training()
    elif st.session_state.view == "Recommendations":
        show_recommendations()
    elif st.session_state.view == "My Ratings":
        show_my_ratings()
    elif st.session_state.view == "Analytics":
        show_analytics()

    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2023 Movie Recommendation System</div>",
                unsafe_allow_html=True)


def show_login_signup():
    """Show the login and signup form."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Login</h2>", unsafe_allow_html=True)
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_username and login_password:
                user_data = load_user_data()

                if login_username in user_data and \
                        user_data[login_username]['password'] == generate_hash(login_password):
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    st.session_state.user_id = user_data[login_username]['user_id']
                    st.session_state.view = "Dashboard"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Sign Up</h2>", unsafe_allow_html=True)
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

        if st.button("Sign Up"):
            if new_username and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    user_data = load_user_data()

                    if new_username in user_data:
                        st.error("Username already exists")
                    else:
                        # Create new user
                        user_id = len(user_data) + 1000  # Start user IDs from 1000
                        user_data[new_username] = {
                            'user_id': user_id,
                            'password': generate_hash(new_password),
                            'created_at': datetime.now().isoformat(),
                            'ratings': []
                        }
                        save_user_data(user_data)

                        st.success("Account created successfully! You can now log in.")
            else:
                st.warning("Please fill in all fields")
        st.markdown("</div>", unsafe_allow_html=True)

    # Demo credentials
    st.info("Demo credentials: Username - 'demo', Password - 'password'")

    # Pre-create demo account if it doesn't exist
    user_data = load_user_data()
    if 'demo' not in user_data:
        user_data['demo'] = {
            'user_id': 1462327,  # Using an ID from the Netflix dataset
            'password': generate_hash('password'),
            'created_at': datetime.now().isoformat(),
            'ratings': []
        }
        save_user_data(user_data)


def show_dashboard():
    """Show the dashboard with system overview."""
    st.markdown("<h2 class='sub-header'>Dashboard</h2>", unsafe_allow_html=True)

    # System status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### System Status")
        if st.session_state.recommender.model:
            st.success("Model: Trained ✓")
        else:
            st.warning("Model: Not trained ⚠")

        if st.session_state.recommender.ratings_data is not None:
            st.success("Ratings data: Loaded ✓")
        else:
            st.warning("Ratings data: Not loaded ⚠")

        if st.session_state.recommender.movie_data is not None:
            st.success("Movie data: Loaded ✓")
        else:
            st.warning("Movie data: Not loaded ⚠")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Algorithm")
        st.info(f"Current algorithm: **{'SVD++' if st.session_state.recommender.use_svdpp else 'SVD'}**")

        if st.session_state.recommender.hyperparams:
            st.write("Hyperparameters:")
            for param, value in st.session_state.recommender.hyperparams.items():
                st.write(f"- {param}: {value}")
        else:
            st.write("Hyperparameters: Default")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Performance Metrics")

        if st.session_state.recommender.metrics['rmse']:
            st.metric("RMSE", f"{st.session_state.recommender.metrics['rmse']:.4f}")
        else:
            st.write("RMSE: Not evaluated")

        if st.session_state.recommender.metrics['mae']:
            st.metric("MAE", f"{st.session_state.recommender.metrics['mae']:.4f}")
        else:
            st.write("MAE: Not evaluated")
        st.markdown("</div>", unsafe_allow_html=True)

    # Quick actions
    st.markdown("<h3 class='sub-header'>Quick Actions</h3>", unsafe_allow_html=True)

    quick_action_col1, quick_action_col2, quick_action_col3 = st.columns(3)

    with quick_action_col1:
        if st.button("🎬 Get Recommendations", key="dash_recommendations"):
            st.session_state.view = "Recommendations"
            st.rerun()

    with quick_action_col2:
        if st.button("⭐ My Ratings", key="dash_ratings"):
            st.session_state.view = "My Ratings"
            st.rerun()

    with quick_action_col3:
        if st.button("📊 View Analytics", key="dash_analytics"):
            st.session_state.view = "Analytics"
            st.rerun()

    # Show popular movies if data is loaded
    if st.session_state.recommender.ratings_data is not None:
        st.markdown("<h3 class='sub-header'>Popular Movies</h3>", unsafe_allow_html=True)

        popular_movies = st.session_state.recommender.get_popular_movies(n=10)

        if len(popular_movies) > 0:
            # Display as a table
            popular_movies = popular_movies.sort_values('Count', ascending=False)

            if 'Name' in popular_movies.columns:
                st.table(popular_movies[['Name', 'Year', 'Count', 'Avg_Rating']].head(10))
            else:
                st.table(popular_movies[['Movie_Id', 'Count', 'Avg_Rating']].head(10))

            # Visualization
            if 'Name' in popular_movies.columns:
                fig = px.bar(
                    popular_movies.head(10),
                    x='Name',
                    y='Count',
                    color='Avg_Rating',
                    color_continuous_scale='RdYlGn',
                    labels={'Count': 'Number of Ratings', 'Name': 'Movie', 'Avg_Rating': 'Average Rating'},
                    title='Most Popular Movies'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No movie popularity data available. Please load ratings data.")
    else:
        st.info(
            "Welcome to the Movie Recommendation System! To get started, go to 'Data Management' to load movie and ratings data.")


def show_data_management():
    """Show the data management section."""
    st.markdown("<h2 class='sub-header'>Data Management</h2>", unsafe_allow_html=True)

    # Data upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Load Movie Data")

    movie_data_option = st.radio(
        "Choose movie data source:",
        ["Upload CSV", "Use sample data"]
    )

    if movie_data_option == "Upload CSV":
        uploaded_movie_file = st.file_uploader(
            "Upload movie data (CSV format with Movie_Id, Year, Name columns)",
            type=["csv"]
        )

        if uploaded_movie_file is not None:
            if st.button("Load Movie Data"):
                with st.spinner("Loading movie data..."):
                    success = st.session_state.recommender.load_movie_data(uploaded_movie_file)
                    if success:
                        st.success(f"Successfully loaded {len(st.session_state.recommender.movie_data)} movies!")
                    else:
                        st.error("Failed to load movie data. Please check the file format.")
    else:
        if st.button("Load Sample Movie Data"):
            # Use a small sample dataset for demonstration
            sample_data = """
            1,2003,Dinosaur Planet
            2,2004,Isle of Man TT 2004 Review
            3,1997,Character
            4,1994,Paula Abdul's Get Up & Dance
            5,2004,The Rise and Fall of ECW
            6,1997,Sick
            7,1992,8 Man
            8,2004,What the #$*! Do We Know!?
            9,1991,Class of Nuke 'Em High 2
            10,2001,Fighter
            """

            # Create a file-like object from the string
            movie_file = io.StringIO(sample_data)

            with st.spinner("Loading sample movie data..."):
                success = st.session_state.recommender.load_movie_data(movie_file)
                if success:
                    st.success(f"Successfully loaded {len(st.session_state.recommender.movie_data)} sample movies!")
                else:
                    st.error("Failed to load sample movie data.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Load Ratings Data")

    ratings_data_option = st.radio(
        "Choose ratings data source:",
        ["Upload files", "Use sample data"]
    )

    if ratings_data_option == "Upload files":
        uploaded_ratings_files = st.file_uploader(
            "Upload ratings files (Netflix format with MovieID: followed by CustomerID,Rating,Date)",
            type=["txt", "csv"],
            accept_multiple_files=True
        )

        if uploaded_ratings_files:
            sample_size = st.number_input(
                "Sample size (leave at 0 for all data):",
                min_value=0,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Larger samples will take longer to process"
            )

            if st.button("Load Ratings Data"):
                progress_bar = st.progress(0)
                with st.spinner("Loading ratings data..."):
                    actual_sample = None if sample_size == 0 else sample_size
                    num_ratings = st.session_state.recommender.load_ratings_data(
                        uploaded_ratings_files, actual_sample, progress_bar
                    )
                    if num_ratings > 0:
                        st.success(f"Successfully loaded {num_ratings} ratings!")
                    else:
                        st.error("Failed to load ratings data. Please check the file format.")
    else:
        if st.button("Load Sample Ratings Data"):
            # Create simple sample ratings data
            sample_data = """
            1:
            1,5,2005-09-06
            2,3,2005-09-06
            3:
            1,4,2005-09-06
            4,3,2005-09-06
            5:
            1,3,2005-09-06
            6,5,2005-09-06
            7:
            1,3,2005-09-06
            2,4,2005-09-06
            9:
            3,4,2005-09-06
            4,5,2005-09-06
            """

            # Split into separate files for testing
            file1 = io.StringIO(sample_data)

            progress_bar = st.progress(0)
            with st.spinner("Loading sample ratings data..."):
                num_ratings = st.session_state.recommender.load_ratings_data(
                    [file1], None, progress_bar
                )
                if num_ratings > 0:
                    st.success(f"Successfully loaded {num_ratings} sample ratings!")
                else:
                    st.error("Failed to load sample ratings data.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Data preview
    if st.session_state.recommender.movie_data is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Movie Data Preview")
        st.dataframe(st.session_state.recommender.movie_data.head(10))
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.recommender.ratings_data is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Ratings Data Preview")
        st.dataframe(st.session_state.recommender.ratings_data.head(10))

        # Ratings distribution
        st.markdown("### Ratings Distribution")
        if hasattr(st.session_state.recommender,
                   'ratings_data') and not st.session_state.recommender.ratings_data.empty:
            if 'Rating' in st.session_state.recommender.ratings_data.columns:
                # Create and display the histogram
                fig = px.histogram(
                    st.session_state.recommender.ratings_data,
                    x='Rating',
                    nbins=5,
                    title='Distribution of Ratings',
                    labels={'Rating': 'Rating Value', 'count': 'Frequency'},
                    color_discrete_sequence=['#1E88E5']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("The 'Rating' column is missing from the loaded data. Cannot display histogram.")
        else:
            st.info("No ratings data loaded yet. Please load data to see the distribution.")
        st.markdown("</div>", unsafe_allow_html=True)

def show_model_training():
    """Show the model training section."""
    st.markdown("<h2 class='sub-header'>Model Training</h2>", unsafe_allow_html=True)

    # Check if data is loaded
    if st.session_state.recommender.data is None:
        st.warning("Please load ratings data in the Data Management section before training the model.")
        if st.button("Go to Data Management"):
            st.session_state.view = "Data Management"
            st.rerun()
        return

    # Model parameters
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Parameters")

    training_mode = st.radio(
        "Choose training mode:",
        ["Quick Training (Default Parameters)", "Hyperparameter Tuning", "Custom Parameters"]
    )

    params = {}

    if training_mode == "Custom Parameters":
        col1, col2 = st.columns(2)

        with col1:
            params['n_factors'] = st.slider("Number of factors:", 20, 200, 100, 10)
            params['n_epochs'] = st.slider("Number of epochs:", 5, 50, 20, 5)

        with col2:
            params['lr_all'] = st.slider("Learning rate:", 0.001, 0.1, 0.005, 0.001, format="%.3f")
            params['reg_all'] = st.slider("Regularization:", 0.01, 0.5, 0.02, 0.01, format="%.2f")
    st.markdown("</div>", unsafe_allow_html=True)

    # Cross-validation
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Cross-Validation")

    cv_enabled = st.checkbox("Enable cross-validation", value=True)

    if cv_enabled:
        k_folds = st.slider("Number of folds:", 2, 10, 5, 1)

        if st.button("Run Cross-Validation"):
            progress_bar = st.progress(0)
            with st.spinner("Performing cross-validation..."):
                cv_results = st.session_state.recommender.cross_validate_model(k_folds, progress_bar)

                if cv_results:
                    # Display cross-validation results
                    st.success(
                        f"Cross-validation complete! Average RMSE: {st.session_state.recommender.metrics['rmse']:.4f}, Average MAE: {st.session_state.recommender.metrics['mae']:.4f}")

                    # Visualize results
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE per Fold", "MAE per Fold"))

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, k_folds + 1)),
                            y=cv_results['test_rmse'],
                            mode='lines+markers',
                            name='RMSE'
                        ),
                        row=1, col=1
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, k_folds + 1)),
                            y=cv_results['test_mae'],
                            mode='lines+markers',
                            name='MAE'
                        ),
                        row=1, col=2
                    )

                    fig.add_hline(
                        y=np.mean(cv_results['test_rmse']),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Avg: {np.mean(cv_results['test_rmse']):.4f}",
                        row=1, col=1
                    )

                    fig.add_hline(
                        y=np.mean(cv_results['test_mae']),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Avg: {np.mean(cv_results['test_mae']):.4f}",
                        row=1, col=2
                    )

                    fig.update_layout(height=400, width=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Hyperparameter tuning
    if training_mode == "Hyperparameter Tuning":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Hyperparameter Tuning")

        if st.button("Start Hyperparameter Tuning"):
            progress_bar = st.progress(0)
            with st.spinner("Performing hyperparameter tuning (this may take a while)..."):
                tuning_results = st.session_state.recommender.tune_hyperparameters(progress_bar)

                if tuning_results is not None:
                    st.success(f"Hyperparameter tuning complete! Best parameters found:")
                    st.json(st.session_state.recommender.hyperparams)

                    # Visualize tuning results
                    fig = px.parallel_coordinates(
                        tuning_results,
                        dimensions=['n_factors', 'n_epochs', 'lr_all', 'reg_all', 'rmse', 'mae'],
                        color='rmse',
                        color_continuous_scale=px.colors.diverging.Tealrose,
                        color_continuous_midpoint=np.mean(tuning_results['rmse'])
                    )
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Train final model
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Train Final Model")

    if st.button("Train Model"):
        progress_bar = st.progress(0)
        with st.spinner("Training model..."):
            # Use appropriate parameters based on training mode
            if training_mode == "Quick Training (Default Parameters)":
                success = st.session_state.recommender.train_model(params={}, progress_bar=progress_bar)
            elif training_mode == "Hyperparameter Tuning":
                success = st.session_state.recommender.train_model(
                    params=st.session_state.recommender.hyperparams,
                    progress_bar=progress_bar
                )
            else:  # Custom Parameters
                success = st.session_state.recommender.train_model(params=params, progress_bar=progress_bar)

            if success:
                st.success("Model trained successfully!")

                # Evaluate the model
                with st.spinner("Evaluating model..."):
                    rmse, mae = st.session_state.recommender.evaluate_model(progress_bar)
                    st.success(f"Model evaluation complete! RMSE: {rmse:.4f}, MAE: {mae:.4f}")

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col2:
                        st.metric("MAE", f"{mae:.4f}")
            else:
                st.error("Failed to train model. Please check your data and parameters.")
    st.markdown("</div>", unsafe_allow_html=True)


def show_recommendations():
    """Show the recommendations section."""
    st.markdown("<h2 class='sub-header'>Movie Recommendations</h2>", unsafe_allow_html=True)

    # Check if model is trained
    if st.session_state.recommender.model is None:
        st.warning("Please train a model in the Model Training section before generating recommendations.")
        if st.button("Go to Model Training"):
            st.session_state.view = "Model Training"
            st.rerun()
        return

    # Recommendation parameters
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Recommendation Settings")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input("User ID:", value=st.session_state.user_id, min_value=1)
        num_recommendations = st.slider("Number of recommendations:", 5, 50, 10)

    with col2:
        min_rating = st.slider("Minimum predicted rating:", 1.0, 5.0, 3.5, 0.5)
        st.write("Algorithm: " + ("SVD++" if st.session_state.recommender.use_svdpp else "SVD"))

    if st.button("Get Recommendations"):
        progress_bar = st.progress(0)
        with st.spinner("Generating recommendations..."):
            recommendations = st.session_state.recommender.get_recommendations(
                user_id, num_recommendations, min_rating, progress_bar
            )

            if recommendations is not None and len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} recommendations for user {user_id}!")

                # Display recommendations
                st.dataframe(recommendations)

                # Visualization
                if len(recommendations) > 0:
                    fig = px.bar(
                        recommendations,
                        x='Name',
                        y='Predicted_Rating',
                        color='Predicted_Rating',
                        color_continuous_scale='RdYlGn',
                        labels={'Predicted_Rating': 'Predicted Rating', 'Name': 'Movie'},
                        title=f'Top Recommendations for User {user_id}'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Download options
                csv = recommendations.to_csv(index=False)
                st.markdown(
                    download_link(recommendations, f"recommendations_user_{user_id}.csv",
                                  "Download Recommendations CSV"),
                    unsafe_allow_html=True
                )
            else:
                st.warning(
                    "No recommendations found that meet the criteria. Try lowering the minimum rating threshold.")
    st.markdown("</div>", unsafe_allow_html=True)

    # User rating history
    if st.session_state.recommender.ratings_data is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### User Rating History")

        user_ratings = st.session_state.recommender.get_user_ratings(user_id)

        if len(user_ratings) > 0:
            st.write(f"User {user_id} has rated {len(user_ratings)} movies:")
            st.dataframe(user_ratings)

            # Visualization
            fig = px.histogram(
                user_ratings,
                x='Rating',
                nbins=5,
                title=f'Rating Distribution for User {user_id}',
                labels={'Rating': 'Rating Value', 'count': 'Number of Movies'},
                color_discrete_sequence=['#FF4B4B']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"User {user_id} has no rating history in the dataset.")
        st.markdown("</div>", unsafe_allow_html=True)


def show_my_ratings():
    """Show the user ratings section."""
    st.markdown("<h2 class='sub-header'>My Ratings</h2>", unsafe_allow_html=True)

    user_id = st.session_state.user_id

    # Add new ratings
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Rate a Movie")

    if st.session_state.recommender.movie_data is None:
        st.warning("Please load movie data in the Data Management section first.")
        return

    # Search for movies
    search_query = st.text_input("Search for a movie:", "")

    if search_query:
        results = st.session_state.recommender.movie_data[
            st.session_state.recommender.movie_data['Name'].str.contains(search_query, case=False, na=False)
        ]

        if len(results) > 0:
            st.write(f"Found {len(results)} movies:")

            for idx, (movie_id, movie) in enumerate(results.iterrows()):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"{movie['Name']} ({movie['Year']})")

                with col2:
                    rating = st.select_slider(
                        f"Rating for {movie['Name']}",
                        options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                        value=3.0,
                        key=f"rating_{movie_id}"
                    )

                with col3:
                    if st.button("Rate", key=f"rate_{movie_id}"):
                        # Add the rating
                        st.session_state.recommender.add_user_rating(user_id, movie_id, rating)

                        # Save to user data
                        user_data = load_user_data()
                        if st.session_state.username in user_data:
                            if 'ratings' not in user_data[st.session_state.username]:
                                user_data[st.session_state.username]['ratings'] = []

                            # Check if movie is already rated
                            for i, r in enumerate(user_data[st.session_state.username]['ratings']):
                                if r['movie_id'] == movie_id:
                                    user_data[st.session_state.username]['ratings'][i]['rating'] = rating
                                    break
                            else:
                                # Add new rating
                                user_data[st.session_state.username]['ratings'].append({
                                    'movie_id': movie_id,
                                    'movie_name': movie['Name'],
                                    'rating': rating,
                                    'timestamp': datetime.now().isoformat()
                                })

                            save_user_data(user_data)

                        st.success(f"Added rating for {movie['Name']}: {rating}!")

                        # Retrain model with new rating
                        if st.session_state.recommender.model is not None:
                            with st.spinner("Updating model with new rating..."):
                                st.session_state.recommender.model.fit(st.session_state.recommender.trainset)
                            st.success("Model updated with new rating!")

                st.markdown("---")
        else:
            st.info(f"No movies found matching '{search_query}'")
    st.markdown("</div>", unsafe_allow_html=True)

    # View and manage ratings
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### My Rating History")

    # Get ratings from user data
    user_data = load_user_data()

    if st.session_state.username in user_data and 'ratings' in user_data[st.session_state.username]:
        user_ratings = user_data[st.session_state.username]['ratings']

        if len(user_ratings) > 0:
            # Convert to DataFrame
            ratings_df = pd.DataFrame(user_ratings)

            # Sort by timestamp (newest first)
            if 'timestamp' in ratings_df.columns:
                ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
                ratings_df = ratings_df.sort_values('timestamp', ascending=False)

            st.write(f"You have rated {len(ratings_df)} movies:")

            # Display as a table
            st.dataframe(ratings_df[['movie_name', 'rating', 'timestamp']])

            # Visualization
            if len(ratings_df) > 0:
                fig = px.histogram(
                    ratings_df,
                    x='rating',
                    nbins=9,
                    title='Your Rating Distribution',
                    labels={'rating': 'Rating Value', 'count': 'Number of Movies'},
                    color_discrete_sequence=['#1E88E5']
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("You haven't rated any movies yet. Use the search box above to find and rate movies.")
    else:
        st.info("You haven't rated any movies yet. Use the search box above to find and rate movies.")
    st.markdown("</div>", unsafe_allow_html=True)


def show_analytics():
    """Show the analytics section."""
    st.markdown("<h2 class='sub-header'>Analytics</h2>", unsafe_allow_html=True)

    # Check if data is loaded
    if st.session_state.recommender.ratings_data is None:
        st.warning("Please load ratings data in the Data Management section to view analytics.")
        return

    # Overview metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        num_users = st.session_state.recommender.ratings_data['Cust_Id'].nunique()
        st.metric("Users", f"{num_users:,}")

    with col2:
        num_movies = st.session_state.recommender.ratings_data['Movie_Id'].nunique()
        st.metric("Movies", f"{num_movies:,}")

    with col3:
        num_ratings = len(st.session_state.recommender.ratings_data)
        st.metric("Ratings", f"{num_ratings:,}")

    with col4:
        density = num_ratings / (num_users * num_movies) * 100
        st.metric("Matrix Density", f"{density:.4f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Rating distribution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Rating Distribution")

    fig = px.histogram(
        st.session_state.recommender.ratings_data,
        x='Rating',
        nbins=9,
        title='Distribution of Ratings',
        labels={'Rating': 'Rating Value', 'count': 'Frequency'},
        color_discrete_sequence=['#FF4B4B']
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # User activity
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### User Activity")

    user_counts = st.session_state.recommender.ratings_data['Cust_Id'].value_counts()

    user_activity_df = pd.DataFrame({
        'User_Id': user_counts.index,
        'Rating_Count': user_counts.values
    }).sort_values('Rating_Count', ascending=False)

    fig = px.box(
        user_activity_df,
        y='Rating_Count',
        title='Distribution of Ratings per User',
        labels={'Rating_Count': 'Number of Ratings'},
        points='all'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top users
    st.write("Top 10 Most Active Users:")
    st.dataframe(user_activity_df.head(10))
    st.markdown("</div>", unsafe_allow_html=True)

    # Movie popularity
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Movie Popularity")

    movie_counts = st.session_state.recommender.ratings_data['Movie_Id'].value_counts()
    movie_avg = st.session_state.recommender.ratings_data.groupby('Movie_Id')['Rating'].mean()

    movie_stats = pd.DataFrame({
        'Movie_Id': movie_counts.index,
        'Rating_Count': movie_counts.values,
        'Avg_Rating': movie_avg.loc[movie_counts.index].values
    }).sort_values('Rating_Count', ascending=False)

    # Merge with movie names if available
    if st.session_state.recommender.movie_data is not None:
        movie_stats = movie_stats.merge(
            st.session_state.recommender.movie_data.reset_index(),
            on='Movie_Id',
            how='left'
        )

    # Scatter plot
    fig = px.scatter(
        movie_stats.head(100),
        x='Rating_Count',
        y='Avg_Rating',
        title='Popularity vs. Average Rating (Top 100 Movies)',
        labels={'Rating_Count': 'Number of Ratings', 'Avg_Rating': 'Average Rating'},
        hover_name='Name' if 'Name' in movie_stats.columns else 'Movie_Id',
        color='Avg_Rating',
        size='Rating_Count',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model performance
    if st.session_state.recommender.metrics['rmse'] is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("RMSE", f"{st.session_state.recommender.metrics['rmse']:.4f}")
            st.write("Root Mean Square Error measures the average magnitude of errors in predictions.")
            st.write("Lower RMSE indicates better accuracy.")

        with col2:
            st.metric("MAE", f"{st.session_state.recommender.metrics['mae']:.4f}")
            st.write(
                "Mean Absolute Error measures the average absolute difference between predictions and actual ratings.")
            st.write("Lower MAE indicates better accuracy.")

        # Cross-validation results
        if st.session_state.recommender.cross_val_results is not None:
            cv_results = st.session_state.recommender.cross_val_results

            fig = make_subplots(rows=1, cols=2, subplot_titles=("RMSE per Fold", "MAE per Fold"))

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cv_results['test_rmse']) + 1)),
                    y=cv_results['test_rmse'],
                    mode='lines+markers',
                    name='RMSE'
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cv_results['test_mae']) + 1)),
                    y=cv_results['test_mae'],
                    mode='lines+markers',
                    name='MAE'
                ),
                row=1, col=2
            )

            fig.add_hline(
                y=np.mean(cv_results['test_rmse']),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {np.mean(cv_results['test_rmse']):.4f}",
                row=1, col=1
            )

            fig.add_hline(
                y=np.mean(cv_results['test_mae']),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {np.mean(cv_results['test_mae']):.4f}",
                row=1, col=2
            )

            fig.update_layout(height=400, width=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Sparsity visualization
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Matrix Sparsity")

    # Create a sample visualization of the sparse matrix
    if st.button("Generate Matrix Sparsity Visualization (Sample)"):
        with st.spinner("Generating sparsity visualization..."):
            # Take a small sample for visualization
            sample_users = np.random.choice(
                st.session_state.recommender.ratings_data['Cust_Id'].unique(),
                size=min(50, num_users),
                replace=False
            )

            sample_movies = np.random.choice(
                st.session_state.recommender.ratings_data['Movie_Id'].unique(),
                size=min(100, num_movies),
                replace=False
            )

            # Filter ratings for the sampled users and movies
            sample_ratings = st.session_state.recommender.ratings_data[
                (st.session_state.recommender.ratings_data['Cust_Id'].isin(sample_users)) &
                (st.session_state.recommender.ratings_data['Movie_Id'].isin(sample_movies))
                ]

            # Create a pivot table
            matrix = pd.pivot_table(
                sample_ratings,
                values='Rating',
                index='Cust_Id',
                columns='Movie_Id',
                fill_value=np.nan
            )

            # Visualize the matrix
            fig = px.imshow(
                matrix.notnull(),
                labels=dict(x="Movie ID", y="User ID", color="Rating"),
                title=f"User-Movie Matrix Sparsity (Sample: {len(sample_users)} users × {len(sample_movies)} movies)",
                color_continuous_scale=["white", "#FF4B4B"]
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate sparsity
            non_zero = matrix.count().sum()
            total = matrix.size
            sparsity = 100 * (1 - non_zero / total)

            st.write(f"Matrix sparsity in this sample: {sparsity:.2f}% (Empty cells)")
    st.markdown("</div>", unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()