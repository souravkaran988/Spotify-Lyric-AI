# 🎵 Spotify Lyric AI: Deep Learning Search Engine

A sophisticated AI-powered search engine capable of identifying songs from snippets of lyrics or artist names.
This project uses **Natural Language Processing (NLP)** and **Deep Learning (TensorFlow)** to understand the semantic meaning of lyrics, allowing users to find songs even if they don't know the exact words.

**Now features Dual-Language Support: English (Global) & Hindi (Bollywood).**

---


## ❓ Problem Statement

Traditional music search engines rely heavily on **keyword matching** (Lexical Search). If a user makes a typo, misremembers a lyric, or provides a partial sentence, standard database queries often fail to return the correct song.

Furthermore, most lyric datasets and search models are biased towards English content, leaving a gap for regional languages like **Hindi**, where users often search using Romanized text (Hinglish).

**The Solution:**
This project replaces rigid keyword matching with **Semantic Vector Search**. By training a Deep Learning model to embed lyrics into a 128-dimensional vector space, the system can understand the *context* of a query. This allows it to identify songs even when the user's input is incomplete, vague, or linguistically mixed.

--- 


## 🚀 Key Features

* **🧠 Context-Aware Lyric Search:** unlike a simple keyword search (Ctrl+F), this AI uses vector embeddings. It can match a user's rough memory of lyrics to the actual song.
* **👤 Intelligent Artist Detection:** The system automatically detects if a query is an artist's name (e.g., "Arijit Singh", "Taylor Swift") and retrieves their entire discography.
* **🌍 Multi-Lingual Support:** Successfully trained on a merged dataset of **60,000+ songs**, covering both International hits and Hindi Bollywood tracks.
* **⚡ Real-Time Performance:** Uses pre-computed vector matrices (`.npy` files) to search through 60,000 songs in milliseconds.
* **🎨 Modern UI:** A responsive, dark-mode web interface built with **Streamlit**.

---

## 🛠️ Technical Architecture (How it Works)

This project is not just a database lookup; it is a Neural Network-based retrieval system.

### 1. The Model Architecture
We use a **TensorFlow/Keras Sequential Model** consisting of:
1.  **TextVectorization Layer:** Converts raw text into integer sequences (Vocabulary size: 10,000 words).
2.  **Embedding Layer:** Maps every word to a dense **128-dimensional vector**. This captures semantic relationships (e.g., "love" and "heart" appear in similar contexts).
3.  **GlobalAveragePooling1D:** Condenses the sequence of word vectors into a single "Song Vector" representing the overall meaning of the track.

### 2. The Search Logic (Cosine Similarity)
1.  When a user types a query, the model converts it into a 128-dimension vector.
2.  We calculate the **Dot Product** (Cosine Similarity) between the *Query Vector* and all *60,000 Song Vectors*.
3.  The system returns the songs with the highest similarity scores (closest angle in vector space).

---

## 📊 Model Accuracy & Performance

The model was evaluated on a held-out test set of **100 random song snippets** (using the first 100 words of the song to mimic the training distribution).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **95.0%** | The correct song was the #1 result. |
| **Top-5 Accuracy** | **100.0%** | The correct song was in the top 5 results. |

---

### Note on Data
- The dataset is compressed as `spotify_songs.zip`.
- Please unzip it to get `spotify_songs.csv` before running the app.

---

## 📂 Project Structure

Spotify-Lyric-AI/
│
├── app.py                # The main Streamlit website script
├── main.ipynb            # The Jupyter Notebook (Training code + Accuracy proof)
├── requirements.txt      # List of libraries (tensorflow, pandas, streamlit)
├── README.md             # The documentation manual (Installation & Features)
├── spotify_songs.csv     # The complete dataset (English + Hindi merged)
│
├── lyric_model.keras     # (Generated) The saved AI Brain
├── song_vectors.npy      # (Generated) The pre-calculated search vectors
└── songs_df.pkl          # (Generated) The fast-access song database

---

## 💻 Installation & Execution Guide

Follow these steps to run the project on your local machine.

### Prerequisites
* Python 3.10 or higher
* VS Code or any Code Editor

### Step 1: Clone the Repository
```bash
- git clone [https://github.com/souravkaran988/Spotify-Lyric-AI.git]
- OR (https://github.com/souravkaran988/Spotify-Lyric-AI.git)

cd Spotify-Lyric-AI

