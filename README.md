

# <div align="center">

# 🌟 **MovieLens Recommender**

### *Discover Your Next Favorite Movie*

A Beautiful, Intelligent, TMDB-Powered Movie Recommendation System

</div>

---

<div align="center">

![Stars](https://img.shields.io/github/stars/navin-oss/MovieRecommendation?style=for-the-badge\&color=8A2BE2)
![Forks](https://img.shields.io/github/forks/navin-oss/MovieRecommendation?style=for-the-badge\&color=9370DB)
![Python](https://img.shields.io/badge/Python-3.9+-blueviolet?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge)
![TMDB API](https://img.shields.io/badge/TMDB-API-green?style=for-the-badge)
![License](https://img.shields.io/github/license/navin-oss/MovieRecommendation?style=for-the-badge)

</div>

## 🎬 Live Demo Screenshots

<div align="center">

### ⭐ Mood-Based Recommendations Preview  
<img src="https://raw.githubusercontent.com/navin-oss/MovieRecommendation/main/Screenshot%202025-12-07%20102609.png" width="850px"/>

<br><br>

### ⭐ Genre Explorer + Recommendations  
<img src="https://raw.githubusercontent.com/navin-oss/MovieRecommendation/main/Screenshot%202025-12-07%20102658.png" width="850px"/>

<br><br>

### ⭐ Full UI Experience  
<img src="https://raw.githubusercontent.com/navin-oss/MovieRecommendation/main/Screenshot%202025-12-07%20102814.png" width="850px"/>

</div>

# 🎯 **Why This Recommender Is Special**

<div align="center">

### 🚀 Smart ML Engine

### 🎞️ Real-Time Posters via TMDB

### 😄 Mood-Based Recommendations

### 🎭 Genre Explorer

### ⚡ Fast, Clean, and Beautiful UI

</div>

---

# ✨ **Key Features (Premium Edition)**

### 🎯 **1. Similar Movie Search**

Pick a movie → Instantly get intelligent ML-powered recommendations
✔ Uses cosine similarity
✔ MovieLens embeddings
✔ Posters from TMDB API

---

### 😄 **2. Mood-Based Suggestions**

Tell the system how you feel —
**Happy**, **Romantic**, **Adventurous**, **Calm**, **Dramatic**, etc.

It returns movies matching your emotional vibe.
Feels magical. ✨

---

### 🎭 **3. Genre Explorer**

Browse hidden gems across 17,990+ genre combinations.
Super fast. Super fun.

---

### 🎞️ **4. TMDB API Integration**

Your app fetches:

* High-quality posters
* Movie descriptions
* Release year
* Ratings
* Genre metadata

No more boring UIs — everything becomes real and visual.

---

### 📊 **5. Quick Stats Panel**

| Metric               | Value      |
| -------------------- | ---------- |
| Movies in Dataset    | **4805**   |
| Genre Combinations   | **17,990** |
| Mood Options         | **7**      |
| Recommendation Paths | **3**      |

---

# 🌈 **Premium UI Preview**

Built with Streamlit + Custom CSS
✔ Dark Theme
✔ Gradient Headers
✔ Animated Buttons
✔ Minimal + Elegant


# 🛠️ **Tech Stack**

<div align="center">

🌐 **Frontend:** Streamlit, HTML/CSS
🧠 **ML Engine:** Python, numpy, pandas, scikit-learn
🗂️ **Data:** MovieLens dataset
🎞️ **API:** TMDB (posters & metadata)
📦 **Storage:** Git LFS for large models

</div>

---

# ⚙️ **Installation**

```bash
git clone https://github.com/navin-oss/MovieRecommendation.git
cd MovieRecommendation
pip install -r requirements.txt
```

---

# 🔐 **TMDB API Key Setup**

Create `.env` file:

```
TMDB_API_KEY=YOUR_KEY_HERE
```

OR use `config.py` based on your project structure.

---

# ▶️ **Run the Application**

```bash
streamlit run app.py
```

---

# 📁 **Project Structure (Ultra Clean)**

```
MovieRecommendation/
│
├── app.py                 # Streamlit UI
├── artifacts/             # similarity.pkl (LFS)
├── data/                  # MovieLens datasets
├── utils/                 # Helper functions
├── assets/                # README images
├── requirements.txt
└── README.md
```

---

# 🧠 How The ML Engine Works

### 1️⃣ **Content-Based Filtering**

* Genre vectors
* Tag features
* Overview embeddings
* Cosine similarity

### 2️⃣ **Mood Intelligence**

* Maps moods → weighted genres
* Curated selections

### 3️⃣ **Genre Explorer**

* Keyword filtering
* Popularity scoring
* Visual recommendations

---

# 🧩 **About `similarity.pkl` (LFS Tracked)**

This file is ~176MB and contains the precomputed similarity matrix.
Git LFS handles:

✔ Upload
✔ Versioning
✔ Delivery

Place it inside:

```
artifacts/similarity.pkl
```

---

# 🚀 **Future Upgrades (Ultra-Premium Roadmap)**

* Deploy on Streamlit Cloud
* TMDB cast/crew integration
* Hybrid (content + collaborative) model
* BERT/SentenceTransformer embeddings
* User login + history-based recommendations
* GIF-based recommendation transitions

---

# ⭐ **Contributions Welcome**

Want to enhance the UI, add new features, or optimize the ML?
PRs are appreciated!

---

# ❤️ **Author**

**Navin**
AIML Engineer • ML Enthusiast
Building beautiful, intelligent apps with clean UI/UX.

---


</div>
