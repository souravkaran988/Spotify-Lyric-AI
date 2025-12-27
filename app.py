# ==========================================
# SPOTIFY LYRIC AI - PRO VERSION (ARTIST + LYRICS)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re

# 1. Page Config
st.set_page_config(page_title="Spotify Lyric AI", page_icon="🎵", layout="wide")

# 2. Styles
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right, #191414, #222831); color: white; }
    h1 { color: #1DB954; font-family: 'Helvetica Neue', sans-serif; text-align: center; }
    .song-card {
        background-color: #282828; padding: 15px; border-radius: 10px;
        margin-bottom: 10px; border-left: 6px solid #1DB954;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .song-title { font-size: 20px; font-weight: bold; color: white; }
    .artist-name { color: #B3B3B3; font-size: 16px; }
    .tag { float: right; background-color: #1DB954; color: black; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 3. Load Data
@st.cache_resource
def load_data():
    model = tf.keras.models.load_model('lyric_model.keras')
    song_vectors = np.load('song_vectors.npy')
    df = pd.read_pickle('songs_df.pkl')
    return model, song_vectors, df

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

def run_search():
    query = st.session_state.search_input
    if not query: return

    model, song_vectors, df = load_data()
    
    # --- LOGIC: ARTIST vs LYRIC SEARCH ---
    
    # 1. Check if query matches an ARTIST name (Case insensitive)
    # We look for partial matches (e.g. "Queen" matches "Queen", "Queen Latifah")
    artist_matches = df[df['artist'].str.contains(query, case=False, na=False)]
    
    if not artist_matches.empty:
        # FOUND ARTIST! Show all their songs
        st.session_state.search_type = "ARTIST"
        st.session_state.results = artist_matches.head(50) # Show top 50 songs
    else:
        # NOT AN ARTIST. Run Vector Search for Lyrics
        st.session_state.search_type = "LYRIC"
        cleaned_query = clean_text(query)
        query_tensor = tf.constant([cleaned_query], dtype=tf.string)
        query_vector = model.predict(query_tensor, verbose=0)
        query_vector = tf.nn.l2_normalize(query_vector, axis=1)
        
        similarities = tf.matmul(query_vector, song_vectors, transpose_b=True)
        top_k_values, top_k_indices = tf.math.top_k(similarities[0], k=10) # Top 10 matches
        
        st.session_state.results = (top_k_indices.numpy(), top_k_values.numpy())
    
    st.session_state.last_query = query

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("🎵 AI Music Master")

    # Initialize State
    if 'results' not in st.session_state: st.session_state.results = None
    if 'search_type' not in st.session_state: st.session_state.search_type = None

    # Search Bar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("Search for an Artist OR Lyrics...", key="search_input", on_change=run_search)
        st.button("🔍 Search", use_container_width=True, on_click=run_search)

    # Display Results
    if st.session_state.results is not None:
        
        # --- MODE 1: ARTIST RESULTS ---
        if st.session_state.search_type == "ARTIST":
            artist_df = st.session_state.results
            st.success(f"🎉 Found {len(artist_df)} songs by '{st.session_state.last_query}'")
            
            for index, row in artist_df.iterrows():
                with st.expander(f"🎶 {row['song']} - {row['artist']}"):
                    st.text(row['text'])
        
        # --- MODE 2: LYRIC RESULTS ---
        else:
            st.info(f"🎤 Searching lyrics for: '{st.session_state.last_query}'")
            indices, scores = st.session_state.results
            _, _, df = load_data()
            
            for i, idx in enumerate(indices):
                song = df.iloc[idx]['song']
                artist = df.iloc[idx]['artist']
                score = scores[i]
                full_lyrics = df.iloc[idx]['text']
                
                st.markdown(f"""
                <div class="song-card">
                    <span class="tag">{score:.0%} MATCH</span>
                    <div class="song-title">{song}</div>
                    <div class="artist-name">👤 {artist}</div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📜 Read Lyrics"):
                    st.text(full_lyrics)

if __name__ == "__main__":
    main()