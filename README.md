# 🎵 Spotify Lyric AI: Deep Learning Search Engine

A sophisticated AI-powered search engine capable of identifying songs from snippets of lyrics or artist names.
This project uses **Natural Language Processing (NLP)** and **Deep Learning (TensorFlow)** to understand the semantic meaning of lyrics, allowing users to find songs even if they don't know the exact words.

**Now features Dual-Language Support: English (Global) & Hindi (Bollywood).**

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
The dataset is compressed as `spotify_songs.zip`.
Please unzip it to get `spotify_songs.csv` before running the app.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | The main **Streamlit** web application script. |
| `main.ipynb` | The **Jupyter Notebook** used for data cleaning, model training, and evaluation. |
| `lyric_model.keras` | The saved **TensorFlow model** (The "Brain"). |
| `songs_df.pkl` | A Pickle file containing the dataframe of all songs (The "Database"). |
| `song_vectors.npy` | A NumPy file containing the pre-computed vectors for all songs (The "Index"). |
| `spotify_songs.csv` | The raw merged dataset (English + Hindi). |
| `requirements.txt` | List of Python dependencies required to run the project. |

---

## 💻 Installation & Execution Guide

Follow these steps to run the project on your local machine.

### Prerequisites
* Python 3.10 or higher
* VS Code or any Code Editor

### Step 1: Clone the Repository
```bash
git clone [https://github.com/souravkaran988/Spotify-Lyric-AI.git](https://github.com/souravkaran988/Spotify-Lyric-AI.git)

cd Spotify-Lyric-AI
