# app.py - Dynamic: content-based + mood-based + genre-based recommendations
import os
import pickle
import requests
import streamlit as st
import pandas as pd
import numpy as np

OMDB_API_KEY = "875ebe98"

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def fetch_movie_details(title: str):
    if not OMDB_API_KEY:
        return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster", "N/A", "N/A"
    try:
        r = requests.get("http://www.omdbapi.com/", params={"t": title, "apikey": OMDB_API_KEY}, timeout=6)
        if r.status_code == 401:
            return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster", "N/A", "N/A"
        r.raise_for_status()
        data = r.json()
        if data.get("Response") == "True":
            poster = data.get("Poster") if data.get("Poster") and data.get("Poster") != "N/A" else "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"
            year = data.get("Year", "N/A")
            imdb_rating = data.get("imdbRating", "N/A")
            return poster, year, imdb_rating
    except:
        pass
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster", "N/A", "N/A"

# ---------------------- Content-based Recommendation ----------------------
def recommend(movie, movies_df, similarity_matrix, top_k=5):
    try:
        index = movies_df[movies_df['title'] == movie].index[0]
    except:
        return [], [], [], []
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    names, posters, years, ratings = [], [], [], []
    for i in distances[1: top_k + 1]:
        title = movies_df.iloc[i[0]].title
        poster, year, rating = fetch_movie_details(title)
        names.append(title)
        posters.append(poster)
        years.append(year)
        ratings.append(rating)
    return names, posters, years, ratings

# ---------------------- Dynamic Mood-Based Recommendation ----------------------
emotion_to_genres = {
    "Happy 😊": ["Comedy", "Family", "Adventure", "Animation", "Musical"],
    "Sad 😢": ["Drama", "Romance", "Music"],
    "Excited 🤩": ["Action", "Thriller", "Sci-Fi", "Adventure", "Crime"],
    "Romantic ❤️": ["Romance", "Drama", "Comedy"],
    "Scared 😨": ["Horror", "Thriller", "Mystery"],
    "Thoughtful 🤔": ["Documentary", "Biography", "Drama", "History", "War"],
    "Adventurous 🚀": ["Adventure", "Fantasy", "Sci-Fi", "Action"]
}

def _row_has_any_genre(cell, genre_list):
    if not cell or pd.isna(cell):
        return False
    text = str(cell).lower()
    for g in genre_list:
        if g.lower() in text:
            return True
    return False

def recommend_by_mood(emotion_display, movies_df, top_k=6):
    genres = emotion_to_genres.get(emotion_display, [])
    if not genres:
        return []
    
    mask = movies_df['genres'].apply(lambda g: _row_has_any_genre(g, genres))
    filtered = movies_df[mask]
    
    if filtered.empty:
        return []
    
    # Try to get diverse movies by sampling from different parts of the dataset
    if len(filtered) > top_k:
        # Sample from beginning, middle, and end to get diversity
        indices = np.linspace(0, len(filtered)-1, top_k, dtype=int)
        sampled = filtered.iloc[indices]
    else:
        sampled = filtered
    
    results = []
    for _, row in sampled.iterrows():
        title = row['title']
        poster, year, rating = fetch_movie_details(title)
        results.append((title, poster, year, rating))
    return results

# ---------------------- Dynamic Genre-Based Recommendation ----------------------
def recommend_by_genre(genre_keyword, movies_df, top_k=6):
    if not genre_keyword:
        return []
    
    # More accurate genre matching
    def genre_match(genre_text, target_genre):
        if not genre_text or pd.isna(genre_text):
            return False
        text = str(genre_text).lower()
        target = target_genre.lower()
        
        # Exact match or contained in genre list (handles pipe-separated genres)
        genres_list = [g.strip().lower() for g in text.split('|')]
        return target in genres_list
    
    mask = movies_df['genres'].apply(lambda g: genre_match(g, genre_keyword))
    filtered = movies_df[mask]
    
    if filtered.empty:
        return []
    
    # Get top_k movies, trying to maintain diversity
    if len(filtered) > top_k:
        # Sample from different parts of the dataset for diversity
        indices = np.linspace(0, len(filtered)-1, top_k, dtype=int)
        sampled = filtered.iloc[indices]
    else:
        sampled = filtered
    
    results = []
    for _, row in sampled.iterrows():
        title = row['title']
        poster, year, rating = fetch_movie_details(title)
        results.append((title, poster, year, rating))
    return results

# ---------------------- Streamlit UI ----------------------
st.set_page_config(layout="wide", page_title="🎬 MovieLens Recommender", initial_sidebar_state="expanded")
st.markdown(
    "<h1 style='text-align:center;color:#fff;background:linear-gradient(90deg,#667eea,#764ba2);padding:10px;border-radius:8px;'>🎬 MovieLens Recommender</h1>",
    unsafe_allow_html=True
)
st.write("")

# ---- Load model/data and normalize genres ----
try:
    movies_dict = pickle.load(open(os.path.join("artifacts", "movie_dict.pkl"), "rb"))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open(os.path.join("artifacts", "similarity.pkl"), "rb"))
except Exception:
    st.error("❌ Model files missing. Put movie_dict.pkl and similarity.pkl into the artifacts/ folder.")
    st.stop()

# Robust creation/normalization of a 'genres' column
if 'genres' not in movies.columns:
    candidates = [c for c in movies.columns if any(k in c.lower() for k in ('genre', 'genres', 'tag', 'category'))]
    if candidates:
        col = candidates[0]
        def normalize_cell(x):
            if x is None:
                return ""
            if isinstance(x, (list, tuple, set)):
                return "|".join(str(i) for i in x)
            return str(x)
        movies['genres'] = movies[col].apply(normalize_cell)
    else:
        movies['genres'] = ""
movies['genres'] = movies['genres'].fillna("").astype(str)

# Build a dynamic list of genre tokens (split by common separators)
def extract_genre_tokens(genres_text):
    if not genres_text:
        return []
    # split by common separators
    parts = []
    for sep in ['|', ',', ';', '/']:
        if sep in genres_text:
            parts = [p.strip() for p in genres_text.split(sep) if p.strip()]
            break
    if not parts:
        parts = [genres_text.strip()] if genres_text.strip() else []
    return [p for p in parts if p and p != 'nan']

all_tokens = set()
for g in movies['genres'].unique():
    for t in extract_genre_tokens(g):
        if t and len(t) > 1:  # Filter out single characters
            all_tokens.add(t)

# Clean and sort genres
all_genres_sorted = sorted([t for t in all_tokens if t and len(t) > 1])

# Fallback: if no tokens found, provide a sensible default list
if not all_genres_sorted:
    all_genres_sorted = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                         "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", 
                         "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Sidebar controls
with st.sidebar:
    st.markdown("### 🔎 Controls")
    top_k = st.slider("Number of Recommendations", 3, 10, 5)
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a Movie", movie_list, help="Choose a movie to find similar recommendations")
    run = st.button("🎯 Get Similar Movies")

    st.markdown("---")
    st.markdown("### 🎭 Mood-Based Suggestions")
    emotions = list(emotion_to_genres.keys())
    selected_emotion = st.selectbox("How are you feeling?", emotions)
    emotion_run = st.button("😊 Get Mood-Based Picks")

    st.markdown("---")
    st.markdown("### 🎬 Genre Explorer")
    selected_genre = st.selectbox("Choose Genre", all_genres_sorted)
    genre_run = st.button("🎭 Discover Genre Gems")

# Main area: content-based
if run:
    with st.spinner("🔍 Finding similar movies..."):
        rec_names, rec_posters, rec_years, rec_ratings = recommend(selected_movie, movies, similarity, top_k)

    if not rec_names:
        st.warning("No recommendations found for the selected movie.")
    else:
        sel_poster, sel_year, sel_rating = fetch_movie_details(selected_movie)
        
        # Display selected movie info
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(sel_poster, width=280)
        with col2:
            st.markdown(f"# {selected_movie}")
            st.markdown(f"### {sel_year} • ⭐ {sel_rating}")
            st.markdown("**Similar Movies You Might Like:**")
        
        st.markdown("---")

        # Recommendation grid
        n_cols = 2 if top_k <= 4 else 3
        cols = st.columns(n_cols)
        for i, (name, poster, year, rating) in enumerate(zip(rec_names, rec_posters, rec_years, rec_ratings)):
            c = cols[i % n_cols]
            with c:
                try:
                    st.image(poster, use_container_width=True)
                except:
                    st.image("https://placehold.co/500x750/333/FFFFFF?text=No+Poster", use_container_width=True)
                st.markdown(f"**{name}**")
                st.caption(f"{year} • ⭐ {rating}")
        
        st.success(f"✅ Found {len(rec_names)} recommendations similar to **{selected_movie}**")
        st.markdown("---")

# Mood-based suggestions (dynamic)
if emotion_run:
    st.markdown(f"## 🎭 Perfect Movies for When You're Feeling {selected_emotion.split()[0].lower()}")
    target_genres = emotion_to_genres.get(selected_emotion, [])
    st.markdown(f"*Curated selection for: **{selected_emotion}***")
    st.markdown(f"*Featuring genres: {', '.join(target_genres)}*")
    
    with st.spinner(f"🎬 Finding perfect {selected_emotion.split()[0].lower()} movies..."):
        emotion_recs = recommend_by_mood(selected_emotion, movies, 6)
    
    if emotion_recs:
        cols = st.columns(3)
        for i, (movie_title, poster, year, rating) in enumerate(emotion_recs):
            col = cols[i % 3]
            with col:
                try:
                    st.image(poster, use_container_width=True)
                except:
                    st.image("https://placehold.co/500x750/333/FFFFFF?text=No+Poster", use_container_width=True)
                st.markdown(f"**{movie_title}**")
                st.caption(f"{year} • ⭐ {rating}")
        
        st.markdown("---")
        st.info("💡 **Pro Tip**: Found a movie you like? Select it from the dropdown above to get more similar recommendations!")
    else:
        st.warning(f"❌ No mood-based suggestions found for '{selected_emotion}'. Try another mood!")

# Genre-based suggestions (dynamic)
if genre_run:
    st.markdown(f"## 🎬 Top {selected_genre} Movies")
    st.markdown(f"*Handpicked selection from the **{selected_genre}** genre*")
    
    with st.spinner(f"🔍 Discovering the best {selected_genre} movies..."):
        genre_recs = recommend_by_genre(selected_genre, movies, 6)
    
    if genre_recs:
        cols = st.columns(3)
        for i, (movie_title, poster, year, rating) in enumerate(genre_recs):
            col = cols[i % 3]
            with col:
                try:
                    st.image(poster, use_container_width=True)
                except:
                    st.image("https://placehold.co/500x750/333/FFFFFF?text=No+Poster", use_container_width=True)
                st.markdown(f"**{movie_title}**")
                st.caption(f"{year} • ⭐ {rating}")
        
        st.markdown("---")
        st.info("💡 **Pro Tip**: Love any of these movies? Use the main search to find similar ones!")
    else:
        st.warning(f"❌ No {selected_genre} movies found in the database. Try another genre!")

# When nothing selected, show instructions
if not run and not emotion_run and not genre_run:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Discover Your Next Favorite Movie
        
        **Three Ways to Find Perfect Movies:**
        
        ### 🎯 **Find Similar Movies**
        - Choose a movie you already love
        - Get intelligent recommendations based on movie content
        - Perfect for when you want "more like this"
        
        ### 🎭 **Match Your Mood** 
        - Tell us how you're feeling right now
        - Get movies that fit your current emotional state
        - Great for when you're not sure what to watch
        
        ### 🎬 **Explore by Genre**
        - Browse your favorite movie categories
        - Discover hidden gems in specific genres
        - Perfect for genre enthusiasts
        
        ---
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Quick Stats
        """)
        st.metric("Movies in Database", len(movies))
        st.metric("Available Genres", len(all_genres_sorted))
        st.metric("Mood Options", len(emotions))
        
        st.markdown("""
        ### ⚡ Quick Start
        1. Pick any option from the sidebar
        2. Click the recommendation button
        3. Explore your personalized results!
        """)