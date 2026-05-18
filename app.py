import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Simple Hybrid Movie Recommender")
st.write("This application combines Content-Based Filtering (Movie Genres) and Collaborative Filtering (User Ratings) to recommend movies.")

# 1. Load Data and Models
@st.cache_resource
def load_data():
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('ratings.pkl', 'rb') as f:
        ratings = pickle.load(f)
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    return movies, ratings, svd_model

try:
    movies, ratings, svd_model = load_data()
    st.success("Models and data loaded successfully!")
except FileNotFoundError:
    st.error("Model files not found! Please run the Jupyter Notebook first to generate the .pkl files.")
    st.stop()

# 2. Define the Hybrid Function
genres_list = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
               "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", 
               "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"]

def get_hybrid_recommendations(user_id, weight_cb, weight_cf, top_n=10):
    # --- Content-Based (CB) ---
    movies['genres_str'] = movies[genres_list].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

    user_ratings = ratings[(ratings["user_id"] == user_id) & (ratings["rating"] >= 4)]
    if len(user_ratings) == 0:
        user_ratings = ratings[ratings["user_id"] == user_id]
        
    movie_ids = user_ratings["movie_id"].values
    
    if len(movie_ids) == 0:
        similarities = np.zeros(len(movies))
    else:
        liked_indices = movies.index[movies["movie_id"].isin(movie_ids)].tolist()
        similarities = cosine_similarity(tfidf_matrix[liked_indices], tfidf_matrix)
        similarities = np.sum(similarities, axis=0)
        if similarities.max() > 0:
            similarities = similarities / similarities.max()

    # --- Collaborative Filtering (CF) ---
    test_set = [(user_id, mid, 0) for mid in movies["movie_id"]]
    expected_ratings = svd_model.test(test_set)
    expected_ratings_pred = np.array([pred.est for pred in expected_ratings])
    cf_scores = expected_ratings_pred / 5.0 # Normalize 1-5 to 0-1

    # --- Hybrid combination ---
    final_scores = (weight_cb * similarities) + (weight_cf * cf_scores)

    # Format output
    result_df = pd.DataFrame({
        "movie_id": movies["movie_id"],
        "title": movies["title"],
        "score": final_scores
    })
    
    # Remove already watched movies
    watched = ratings[ratings["user_id"] == user_id]["movie_id"].values
    result_df = result_df[~result_df["movie_id"].isin(watched)]
    
    return result_df.sort_values(by="score", ascending=False).head(top_n)

# 3. User Inputs
st.sidebar.header("Settings")
user_list = sorted(ratings['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_list)

st.sidebar.subheader("Model Weights")
weight_cf = st.sidebar.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7, 0.1)
weight_cb = 1.0 - weight_cf
st.sidebar.write(f"Content-Based Weight: **{weight_cb:.1f}**")

num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# 4. Display Recommendations
if st.button("Show Recommendations"):
    st.subheader(f"Top {num_recs} Recommendations for User {selected_user}")
    
    with st.spinner("Calculating recommendations..."):
        recommendations = get_hybrid_recommendations(selected_user, weight_cb, weight_cf, num_recs)
        
        # Display as a clean table
        st.table(recommendations[['title', 'score']].reset_index(drop=True))
