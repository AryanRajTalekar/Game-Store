import streamlit as st
import pandas as pd
import numpy as np
import os
os.system("pip install scikit-learn")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import random



# Configure the app (EXACTLY AS PROVIDED)
st.set_page_config(
    layout="wide",
    page_title="GameVerse - Discover & Play",
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# Custom CSS (EXACTLY AS PROVIDED - NO CHANGES)
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .header1 {
        color: #FF4B4B;
        text-align: left;
        font-size: 3em;
        margin-bottom: 0.5em;   
    }
    .subheader {
        color: #1F77B4;
        font-size: 1.5em;
        margin-top: 1em;
    }
    .subheader {
        color: #1F77B4;
        font-size: 1.5em;
        margin-top: 1em;
    }
    .game-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        background-color: #f9f9f9;
    }
    .stTabs [data-baseweb="tab"] {
        width: 100%;
        height: 50px;
        padding: 0 25px;
        background-color: black;
        border-radius: 10px 10px 0 0;
        border: 1px solid #2E4053;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .game-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .feature-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin: 0.5em 0;
    }
    .stSelectbox>div>div>select {
        padding: 8px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess game data"""
    try:
        # Replace with your actual data source
        df = pd.read_csv("cleaned_data.csv")
        df = df.head(500)
        # Data cleaning
        df = df.dropna(subset=["Name", "Icon URL"])
        df["Average User Rating"] = pd.to_numeric(df["Average User Rating"], errors="coerce").fillna(0)
        df["User Rating Count"] = pd.to_numeric(df["User Rating Count"], errors="coerce").fillna(0)

        # Create combined features for recommendation
        df["features"] = (
            df["Genres"] + " " +
            df["Primary Genre"] + " " +
            df["Description"].fillna("") + " " +
            df["Developer"].fillna("") + " " +
            df["Average User Rating"].astype(str) + " " +
            df["User Rating Count"].astype(str)
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def build_recommendation_model(df):
    """Build TF-IDF model and similarity matrix"""
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(df["features"])
        cosine_sim = cosine_similarity(tfidf_matrix)
        return vectorizer, cosine_sim

    except Exception as e:
        st.error(f"Error building model: {str(e)}")
        return None, None

def display_game_card(game, col=None, show_similarity=False, similarity_score=None):
    """Display game card (EXACTLY AS PROVIDED)"""
    if col is None:
        col = st
    
    with col:
        st.image(
            game['Icon URL'] if pd.notna(game['Icon URL']) else 'https://via.placeholder.com/150',
            width=150
        )
        st.write(f"**{game['Name']}**")
        rating = game['Average User Rating']
        stars = '⭐' * int(round(rating)) + '☆' * (5 - int(round(rating)))
        st.write(f"{stars} {rating:.1f}")
        st.caption(f"{int(game['User Rating Count']/1000)}K reviews • {game['Primary Genre']}")
        if show_similarity and similarity_score:
            st.progress(float(similarity_score))
            st.caption(f"Match: {similarity_score:.0%}")

def show_featured_games(df):
    """Display featured games (MODIFIED TO SHOW 10 GAMES PER GENRE)"""
    st.markdown("<h2 class='header1'>Suggested For You:</h2>", unsafe_allow_html=True)
    st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)
    
    # Select 5 random genres
    genres = df['Primary Genre'].dropna().unique()
    selected_genres = list(genres)
    
    for genre in selected_genres:
        st.markdown(f"<h3 class='subheader'>{genre}</h3>", unsafe_allow_html=True)
        
        # Get top 10 games in this genre (was 5)
        genre_games = df[df['Primary Genre'] == genre]
        genre_games = genre_games.sort_values(
            by=['Average User Rating', 'User Rating Count'], 
            ascending=False
        ).head(10)
        
        # Display in two rows of 5
        cols = st.columns(5)
        for i, (_, game) in enumerate(genre_games[:5].iterrows()):
            display_game_card(game, cols[i])
        
        if len(genre_games) > 5:
            cols = st.columns(5)
            for i, (_, game) in enumerate(genre_games[5:10].iterrows()):
                display_game_card(game, cols[i])
                
            st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 2px; 
            margin: 20px 0;'>
</div>
""", unsafe_allow_html=True)

def show_recommendation_section(df, vectorizer, cosine_sim):
    """Display game recommendation section (MODIFIED FOR 20 GAMES)"""
    st.markdown("""
<h1 class='header' style='background: linear-gradient(45deg, #FF4B4B, #1F77B4);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
text-align:left'>
🔍 Find Your Next Favorite Game
</h1>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Based on a Game You Like", "✨ Custom Preferences"])

    with tab1:
        st.markdown("<h3 class='subheader'>Find similar games</h3>", unsafe_allow_html=True)
        selected_game = st.selectbox(
            "Select a game you enjoy:",
            df["Name"].sort_values().unique(),
            key="game_select"
        )

        if st.button("Find Similar Games", key="similar_btn"):
            try:
                game_idx = df[df["Name"] == selected_game].index[0]
                sim_scores = list(enumerate(cosine_sim[game_idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]  # Now 20 games

                st.markdown(f"<h4>Games similar to <span style='color:#FF4B4B'>{selected_game}</span>:</h4>", unsafe_allow_html=True)

                # Display in 4 rows of 5
                for i in range(0, 20, 5):
                    
                    cols = st.columns(5)
                    for j in range(5):
                        if i+j < len(sim_scores):
                            idx, score = sim_scores[i+j]
                            display_game_card(df.iloc[idx], cols[j], show_similarity=True, similarity_score=score)
                    st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("<h3 class='subheader'>Tell us what you like</h3>", unsafe_allow_html=True)

        with st.form("preferences_form"):
            col1, col2 = st.columns(2)

            with col1:
                genres = st.text_input("Favorite genres (comma separated)", "Adventure, Action")
                min_rating = st.slider("Minimum rating", 0.0, 5.0, 3.5, 0.1)

            with col2:
                keywords = st.text_input("Keywords you like in games", "explore, puzzle")
                min_reviews = st.number_input("Minimum reviews", min_value=0, value=100)

            submitted = st.form_submit_button("Find Games For Me")
            st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)

            if submitted:
                try:
                    query = f"{genres} {keywords}"
                    query_vector = vectorizer.transform([query])

                    sim_scores = cosine_similarity(query_vector, vectorizer.transform(df["features"]))
                    sim_scores = list(enumerate(sim_scores[0]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                    filtered = []
                    for idx, score in sim_scores:
                        game = df.iloc[idx]
                        if (game['Average User Rating'] >= min_rating and
                            game['User Rating Count'] >= min_reviews):
                            filtered.append((idx, score))

                    if not filtered:
                        st.warning("No games match all your criteria. Try relaxing your filters.")
                    else:
                        st.markdown("<h4>Recommended games for you:</h4>", unsafe_allow_html=True)

                        # Show top 20 matches (was 5)
                        for i in range(0, min(20, len(filtered)), 5):
                            
                            cols = st.columns(5)
                            for j in range(5):
                                if i+j < min(20, len(filtered)):
                                    idx, score = filtered[i+j]
                                    display_game_card(df.iloc[idx], cols[j], show_similarity=True, similarity_score=score)
                            st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def show_game_browser(df):
    """Display game browser section (MODIFIED TO SHOW 20 GAMES)"""
    st.markdown("""
<h1 class='header' style='background: linear-gradient(45deg, #FF4B4B, #1F77B4);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
DISCOVER YOUR NEXT ADVENTURE
</h1>
""", unsafe_allow_html=True)

    # Filters (EXACTLY AS PROVIDED)
    with st.expander("🔍 Filter Games", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            genre_filter = st.selectbox(
                "Genre",
                ["All"] + sorted(df["Primary Genre"].unique().tolist()),
                key="genre_filter"
            )

        with col2:
            rating_filter = st.slider(
                "Minimum Rating",
                0.0, 5.0, 3.0,
                key="rating_filter"
            )

        with col3:
            reviews_filter = st.number_input(
                "Minimum Reviews",
                min_value=0,
                value=100,
                key="reviews_filter"
            )

    # Apply filters
    filtered_df = df.copy()
    if genre_filter != "All":
        filtered_df = filtered_df[filtered_df["Primary Genre"] == genre_filter]
    filtered_df = filtered_df[filtered_df["Average User Rating"] >= rating_filter]
    filtered_df = filtered_df[filtered_df["User Rating Count"] >= reviews_filter]

    # Sort options
    sort_option = st.selectbox(
        "Sort by",
        ["Rating (High to Low)", "Reviews (High to Low)", "Alphabetical"],
        key="sort_option"
    )

    if sort_option == "Rating (High to Low)":
        filtered_df = filtered_df.sort_values("Average User Rating", ascending=False)
    elif sort_option == "Reviews (High to Low)":
        filtered_df = filtered_df.sort_values("User Rating Count", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("Name")

    # Display 20 games (was showing all)
    st.markdown(f"<h4>Showing {len(filtered_df)} games (top 20):</h4>", unsafe_allow_html=True)
    
    for i in range(0, min(20, len(filtered_df)), 5):
        cols = st.columns(5)
        for j in range(5):
            if i+j < min(20, len(filtered_df)):
                display_game_card(filtered_df.iloc[i+j], cols[j])
        st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)

def main():
    # Load data and build model (EXACTLY AS PROVIDED)
    df = load_data()
    if df.empty:
        st.error("Failed to load game data. Please check your data file.")
        return
    
    vectorizer, cosine_sim = build_recommendation_model(df)
    if vectorizer is None:
        st.error("Failed to build recommendation model.")
        return
    
    # Sidebar (EXACTLY AS PROVIDED)
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50.png?text=GameVerse", width=150)
        st.markdown("<h2>Discover New Games</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p>Find your next favorite game based on:</p>
        <ul>
            <li>Games you already like</li>
            <li>Your preferred genres</li>
            <li>Game features you enjoy</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<small>contact</small>", unsafe_allow_html=True)
        
    # Main content with tabs (EXACTLY AS PROVIDED)
    
    


   
    st.markdown("""
<div style='
    background: linear-gradient(90deg, rgba(255,75,75,0.7), rgba(31,119,180,0.7));
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
'>
    <h1 style='
        color: white;
        text-align: center;
        font-size: 3em;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    '>🎮 GAMEVERSE 🕹️</h1>
    
</div>
                
""", unsafe_allow_html=True)
    st.markdown("""
<div style='height: 4px; 
            background: linear-gradient(90deg, #FF4B4B, #1F77B4);
            border-radius: 4px; 
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
</div>
""", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Featured Games", "Top Games", "Recommendations"])
    
    with tab1:
        show_featured_games(df)
    with tab2:
        show_game_browser(df)
    with tab3:
        show_recommendation_section(df, vectorizer, cosine_sim)

if __name__ == "__main__":
    main()