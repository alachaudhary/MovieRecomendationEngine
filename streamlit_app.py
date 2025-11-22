# ============================================================================
# STREAMLIT MOVIE RECOMMENDATION APPLICATION
# Item-Based Collaborative Filtering using KNN
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gdown


try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.neighbors import NearestNeighbors

# ============================================================================
# PAGE CONFIG (LIGHT MODE UI)
# ============================================================================
st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM LIGHT THEME UI STYLE
# ============================================================================
st.markdown("""
<style>
    body { background-color: #ffffff !important; }
    .stTextInput>div>div>input {
        background-color: #fafafa;
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
# ============================================================================ 
# LOAD MODELS (with auto-download from Google Drive)
# ============================================================================ 

@st.cache_resource
def load_models():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Map of local filenames → Google Drive shareable URLs
    files_to_download = {
        "knn_model.pkl": "https://drive.google.com/uc?id=1-Eu1-tS8knPEpbycDClEe-o3D5WIXUyx",
        "movie_to_idx.pkl": "https://drive.google.com/uc?id=14zpPD906VkCVmv-FY97yVRI4VYtyzTDN",
        "user_to_idx.pkl": "https://drive.google.com/uc?id=1swRrIFWvhWstQKO3cTRmtzI3VJ5S5HX8",
        "sparse_matrix.pkl": "https://drive.google.com/uc?id=19WR7WB3jgaZt74k8CudC4rkNwoYVaXFh",
        "movies_processed.csv": "https://drive.google.com/uc?id=1kbkn4WW7_X8Ywv0AmnpvtyYnbzg_lr0h",
        "ratings_processed.csv": "https://drive.google.com/uc?id=1URBhoGbZSZd7q5ZEfQ_EaUE8o3wsVm7e"
    }

   # Download missing files
    for filename, url in files_to_download.items():
        file_path = model_dir / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            st.info(f"Downloading {filename}...")
            gdown.download(url, str(file_path), quiet=False)

    # Load models safely
    with open(model_dir / "knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(model_dir / "movie_to_idx.pkl", "rb") as f:
        movie_to_idx = pickle.load(f)
    with open(model_dir / "user_to_idx.pkl", "rb") as f:
        user_to_idx = pickle.load(f)
    with open(model_dir / "sparse_matrix.pkl", "rb") as f:
        sparse_matrix = pickle.load(f)

    movies = pd.read_csv(model_dir / "movies_processed.csv")
    
    return knn, movie_to_idx, user_to_idx, sparse_matrix, movies


# ============================================================================
# RECOMMENDATION LOGIC
# ============================================================================
def get_recommendations(movie_name, knn, movie_to_idx, sparse_matrix, movies, n=10):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if len(movie_list) == 0:
        return None, None, []

    matching = movie_list['title'].tolist()
    movie = movie_list.iloc[0]
    movie_id = movie['movieId']

    if movie_id not in movie_to_idx:
        return None, None, matching

    movie_idx = movie_to_idx[movie_id]

    distances, indices = knn.kneighbors(sparse_matrix[movie_idx], n_neighbors=n+1)

    recs = []
    movie_ids_list = list(movie_to_idx.keys())

    for idx, dist in zip(indices.squeeze().tolist()[1:], distances.squeeze().tolist()[1:]):
        mid = movie_ids_list[idx]
        rec_movie = movies[movies['movieId'] == mid]
        if len(rec_movie) > 0:
            r = rec_movie.iloc[0]
            similarity = 1 - dist
            recs.append({
                "Movie Title": r["title"],
                "Genres": r["genres"],
                "Similarity Score": f"{similarity:.4f}"
            })

    df = pd.DataFrame(recs)
    return (movie["title"], movie["genres"]), df, matching


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style='text-align:center; padding:2rem;'>
    <h1 style="color:#3a86ff;">🎬 Movie Recommendation System</h1>
    <p style="font-size:18px; color:#666;">
        Discover movies similar to your favorite using Machine Learning (KNN)
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### ⏳ Loading Models...")
with st.spinner("Loading machine learning model..."):
    knn, movie_to_idx, user_to_idx, sparse_matrix, movies = load_models()
st.success("Model loaded successfully!")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 System Information")
    col1, col2 = st.columns(2)
    col1.metric("🎬 Movies", len(movies))
    col2.metric("👥 Users", len(user_to_idx))
    
   
    st.markdown("---")
    st.markdown("### 🔧 Settings")
    n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)

    st.markdown("---")
    
    st.markdown("### ℹ️ How It Works")
    st.info("""
    **Item-Based Collaborative Filtering:**
    
    1. You search for a movie
    2. System finds similar movies based on user ratings
    3. Returns top recommendations with similarity scores
    
    **Similarity Metric:** Cosine Distance (measures angle between rating vectors)
    """)
 


# ============================================================================
# SEARCH BAR
# ============================================================================
st.markdown("### 🔍 Search for a Movie")
query = st.text_input("Enter movie name:", placeholder="e.g., Inception, Titanic, Avatar...")

if query:
    st.markdown("---")
    with st.spinner(f"Finding movies similar to '{query}'..."):
        movie_info, rec_df, matching = get_recommendations(
            query, knn, movie_to_idx, sparse_matrix, movies, n_recommendations
        )

    if movie_info is None:
        st.error("Movie not found in dataset.")
    else:
        title, genres = movie_info

        st.markdown(f"""
            <div style='background:#eef4ff; padding:1.5rem; border-radius:12px;'>
                <h3 style='color:#3a86ff;'>🎯 Selected Movie</h3>
                <h2>{title}</h2>
                <p><strong>Genres:</strong> {genres}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### 📽️ Top {len(rec_df)} Recommendations")

        cols = st.columns(2)
        for i, row in rec_df.iterrows():
            similarity = float(row["Similarity Score"])
            if similarity >= 0.7:
                color, badge = "#d4f8d4", "🔥 Highly Similar"
            elif similarity >= 0.5:
                color, badge = "#ffeec9", "✨ Similar"
            else:
                color, badge = "#f0e6ff", "🎬 Related"

            with cols[i % 2]:
                st.markdown(f"""
                    <div style='background:{color}; padding:1.2rem; border-radius:10px; border-left:5px solid #3a86ff; margin-bottom:1rem;'>
                        <p style='font-size:12px; color:#555;'>{badge}</p>
                        <h4>{row["Movie Title"]}</h4>
                        <p style='font-size:12px;'><strong>Genres:</strong> {row["Genres"]}</p>
                        <p style='font-size:14px; font-weight:bold; color:#3a86ff;'>Similarity: {row["Similarity Score"]}</p>
                    </div>
                """, unsafe_allow_html=True)


# ============================================================================
# 📘 ACADEMIC PROJECT DESCRIPTION
# ============================================================================
st.markdown("""
<br><br>
<div style="
    background:#f5f7fa;
    padding:2rem;
    border-radius:12px;
    border-left:6px solid #3a86ff;
">
    <h2 style="color:#3a86ff;">🎓 Academic Project Description</h2>
    <p style="font-size:16px; color:#444;">
        This Movie Recommendation System is developed as part of the coursework at the 
        <strong>University of Management and Technology (UMT), Lahore</strong>.
        The objective is to build an end-to-end Machine Learning project demonstrating 
        data preprocessing, sparse matrix handling, model training, and real-time recommendations.
    </p>
    <p style="font-size:15px; color:#444;">
        <strong>Course:</strong> Data Science Technologies (CS4135)<br>
        <strong>Supervisor:</strong> Sir Muhammad Abdul Jabbar<br>
        <strong>Student:</strong> Abd-ul-ala Taha<br>
        <strong>Model:</strong> Item-Based Collaborative Filtering (KNN)<br>
        <strong>Dataset:</strong> MovieLens Ratings Dataset
    </p>
</div>
<br>
""", unsafe_allow_html=True)


# ============================================================================
# 🧹 DATA PREPROCESSING
# ============================================================================
st.markdown("""
<div style="background:#ffffff; padding:2rem; border-radius:12px; border:1.5px solid #e3e6ea;">
    <h2 style="color:#3a86ff;">🧹 Data Preprocessing</h2>

    <p>Before training the recommendation model, the MovieLens dataset underwent careful preprocessing to ensure data quality and improve model performance.</p>

    <h3>1. Dataset Overview</h3>
    <ul>
        <li><strong>Movies Dataset:</strong> 86,537 movies with columns <em>movieId, title, genres</em></li>
        <li><strong>Ratings Dataset:</strong> 33,832,162 ratings from 330,975 users with columns <em>userId, movieId, rating, timestamp</em></li>
        <li><strong>Rating Range:</strong> 0.5 to 5.0, with an average rating of 3.54</li>
    </ul>

    <h3>2. Handling Missing Values</h3>
    <ul>
        <li>No missing values were found in either movies or ratings datasets, ensuring completeness of data</li>
    </ul>

    <h3>3. Removing Duplicates</h3>
    <ul>
        <li>Duplicate ratings were checked and removed (0 duplicates found)</li>
        <li>Remaining ratings after deduplication: 33,832,162</li>
    </ul>

    <h3>4. Filtering Low-Activity Movies and Users</h3>
    <ul>
        <li>Movies with fewer than 10 ratings were removed to ensure sufficient data for similarity computation:
            <br>Before filtering: 83,239 movies → After filtering: 32,021 movies</li>
        <li>Users with fewer than 30 ratings were removed to focus on active users:
            <br>Before filtering: 330,895 users → After filtering: 169,521 users</li>
    </ul>

    <h3>5. Final Dataset</h3>
    <ul>
        <li><strong>Movies:</strong> 32,021</li>
        <li><strong>Users:</strong> 169,521</li>
        <li><strong>Ratings:</strong> 31,502,217</li>
    </ul>

    <p>These preprocessing steps ensure that the dataset is clean, consistent, and well-suited for building a reliable movie recommendation model using K-Nearest Neighbors.</p>
</div>
<br>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<hr>
<div style='text-align:center; color:#666; padding:1rem;'>
    <p>Built by Abd-ul-ala | Item-Based Collaborative Filtering | KNN Algorithm</p>
    <p style='font-size:12px;'>Movie Recommendation System © 2023</p>
</div>
""", unsafe_allow_html=True)
