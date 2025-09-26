# MIVIE-RF: Movie Recommendation System

Welcome to **MIVIE-RF** – a content-based movie recommendation system leveraging the power of the Random Forest algorithm. Built using **Python** and **Flask**, this project demonstrates how machine learning can enhance user experience by providing personalized movie suggestions.

---

## 📽️ About the Project

**MIVIE-RF** aims to recommend movies to users based on the content features of movies they've liked or interacted with. Unlike collaborative filtering, which relies on user-user or item-item similarities, content-based filtering analyzes the characteristics (genres, actors, directors, keywords, etc.) of movies to find similar ones.

The project utilizes the **Random Forest** algorithm to model the relationship between movie features and user preferences, yielding intelligent and relevant recommendations.

---

## 🚀 Features

- **Content-Based Filtering:** Recommends movies based on their attributes and user history.
- **Random Forest Algorithm:** Employs ensemble learning for robust prediction of user preferences.
- **Python & Flask Backend:** RESTful API for easy integration and extensibility.
- **User-Friendly Interface:** Simple endpoints for fetching recommendations.

---

## 🧠 How Does the Random Forest Algorithm Work?

Random Forest is an ensemble machine learning algorithm that builds multiple decision trees and merges their results to improve accuracy and avoid overfitting.

**Key Steps:**
1. **Data Preparation:** Movie features (genres, actors, director, etc.) are encoded into numerical vectors.
2. **Training:** Multiple decision trees are trained on random subsets of the data and features.
3. **Prediction:** Each tree votes for its recommendation; the movies with the most votes are suggested to the user.
4. **Aggregation:** The final recommendation is derived by aggregating (majority voting or averaging) the predictions from all the trees.

**Why Random Forest?**
- Handles high-dimensional data well (many movie attributes).
- Reduces risk of overfitting compared to a single decision tree.
- Robust to noisy data.

> _**In this project, Random Forest helps in modeling complex interactions between movie features and user preferences, leading to highly relevant recommendations.**_


![Random Forest GIF](randomforest.gif)
---

## 🛠️ Tech Stack

- **Python:** Core programming language for backend and ML logic.
- **Flask:** Lightweight web framework to expose RESTful APIs.
- **scikit-learn:** For implementing the Random Forest algorithm.
- **pandas, numpy:** Data manipulation and preprocessing.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/janvi0303/MIVIE-RF.git
   cd MIVIE-RF
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```bash
   python app.py
   ```

4. **Access the API:**
   - Default: `http://127.0.0.1:5000/`

---

## 🕹️ Usage

- **Get Recommendations:**
  Send a POST request to `/recommend` with user preferences (liked movies or feature vector).
  ```json
  {
    "liked_movies": ["Inception", "The Matrix"]
  }
  ```

- **Response:**
  ```json
  {
    "recommended": ["Interstellar", "Minority Report", ...]
  }
  ```

---

## 📑 Example Code Snippet

```python
from sklearn.ensemble import RandomForestClassifier

# Assume X_train is feature matrix, y_train is user preference label
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict preference for new movies
predictions = model.predict(X_new)
```

---

## 👩‍💻 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---

## 🙋‍♀️ Contact

Created by [janvi0303](https://github.com/janvi0303) – feel free to reach out for questions or collaborations!
