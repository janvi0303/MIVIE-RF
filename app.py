from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Merge datasets on the 'title' column
movies = movies.merge(credits[['movie_id', 'title', 'cast', 'crew']], on='title')

# Select relevant columns and handle missing data
movies = movies[['movie_id', 'title', 'cast', 'crew', 'genres', 'popularity', 'vote_average', 'vote_count']]
movies.dropna(inplace=True)

# Feature Engineering - Convert categorical data into numbers for the model
def convert_to_list(string):
    if isinstance(string, str):
        return string.replace("'", '').replace("[", '').replace("]", '').split(',')
    return []

movies['cast'] = movies['cast'].apply(convert_to_list)
movies['crew'] = movies['crew'].apply(convert_to_list)
movies['genres'] = movies['genres'].apply(convert_to_list)

# We'll convert lists into string for vectorization
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x[:3]))  # Limit to 3 main actors
movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x[:2]))  # Limit to 2 main crew members
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))  # All genres

# Combine the text features for vectorization
movies['combined_features'] = movies['cast'] + ' ' + movies['crew'] + ' ' + movies['genres']

# Prepare the final dataset by combining features and labels
features = movies[['combined_features', 'popularity', 'vote_average', 'vote_count']]
X = pd.get_dummies(features, drop_first=True)  # One-hot encoding
y = movies['popularity']  # Target variable

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Recommendation function based on a selected movie
def recommend_movies(movie_title):
    movie = movies[movies['title'].str.lower() == movie_title.lower()]

    if movie.empty:
        return []

    # Get the index of the selected movie
    movie_index = movie.index[0]

    # Get the combined features for the selected movie
    movie_features = X.iloc[movie_index].values.reshape(1, -1)

    # Calculate the cosine similarity between the selected movie and all other movies
    similarity = cosine_similarity(movie_features, X)

    # Get the indices of the most similar movies (excluding the selected movie)
    similar_indices = np.argsort(similarity[0])[-6:-1]  # Get top 5 similar movies (excluding the movie itself)

    # Get the titles of the recommended movies
    recommended_movies = movies.iloc[similar_indices]['title'].tolist()

    return recommended_movies

# Flask routes
@app.route('/')
def home():
    movie_titles = movies['title'].tolist()  # Get a list of all movie titles
    return render_template('index.html', movies=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = recommend_movies(movie_title)
    return render_template('recommendations.html', movies=recommendations)

if __name__ == "__main__":
    app.run(debug=True)