from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# LOAD FILES
# =========================

df      = joblib.load("movies.pkl")
vectors = joblib.load("vectors.pkl")
tfidf   = joblib.load("tfidf.pkl")

# =========================
# CONFIG
# =========================

GENRES = [
    "Action", "Comedy", "Drama", "Horror",
    "Romance", "Thriller", "Animation", "Crime",
    "Science Fiction", "Fantasy", "Documentary", "Adventure",
]

# =========================
# HELPERS
# =========================

def build_poster_url(poster_path):
    if pd.notna(poster_path) and str(poster_path).strip():
        val = str(poster_path).strip()
        if val in ("", "nan", "None"):
            return ""
        if val.startswith("http"):
            return val
        if val.startswith("/"):
            return "https://image.tmdb.org/t/p/w500" + val
    return ""


def get_genres():
    return GENRES


def row_to_card(idx):
    row = df.iloc[idx]
    return {
        "id":       int(idx),
        "title":    row["original_title"],
        "genre":    row.get("genre", ""),
        "overview": row.get("overview", ""),
        "poster":   build_poster_url(row.get("poster_path", "")),
    }

# =========================
# RECOMMENDATION FUNCTION
# =========================

def recommend(movie_name, n=6):
    movie_name = movie_name.lower().strip()

    matches = df[df["original_title"].str.lower() == movie_name]
    if matches.empty:
        matches = df[df["original_title"].str.lower().str.contains(movie_name, regex=False)]

    if matches.empty:
        return None, []

    movie_index   = matches.index[0]
    searched_card = row_to_card(movie_index)

    movie_vector = vectors[movie_index]
    scores  = cosine_similarity(movie_vector, vectors).flatten()
    similar = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, _ in similar[1:]:
        card = row_to_card(idx)
        if card["poster"]:
            results.append(card)
        if len(results) == n:
            break

    return searched_card, results

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    cards = [row_to_card(i) for i in df.head(24).index]
    return render_template(
        "index.html",
        movies=cards,
        genres=get_genres(),
        current_offset=24,
        selected_genre="",
    )


@app.route("/genre/<genre>")
def genre_movies(genre):
    filtered = df[df["genre"].str.contains(genre, case=False, na=False)]
    cards    = [row_to_card(i) for i in filtered.head(24).index]
    return render_template(
        "index.html",
        movies=cards,
        genres=get_genres(),
        selected_genre=genre,
        current_offset=24,
    )


@app.route("/load-more")
def load_more():
    genre  = request.args.get("genre", "").strip()
    offset = int(request.args.get("offset", 0))
    limit  = 24

    if genre:
        filtered = df[df["genre"].str.contains(genre, case=False, na=False)]
    else:
        filtered = df

    total    = len(filtered)
    chunk    = filtered.iloc[offset: offset + limit]
    cards    = [row_to_card(i) for i in chunk.index]
    has_more = (offset + limit) < total

    return jsonify({"movies": cards, "has_more": has_more})


@app.route("/autocomplete")
def autocomplete():
    q = request.args.get("q", "").lower().strip()
    if len(q) < 2:
        return jsonify([])
    matches = df[df["original_title"].str.lower().str.contains(q, regex=False)]
    titles  = matches["original_title"].head(8).tolist()
    return jsonify(titles)


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    movie_name = request.form.get("movie", "").strip()

    searched_card, recommendations = recommend(movie_name)

    return render_template(
        "index.html",
        searched_card=searched_card,
        recommendations=recommendations,
        searched_movie=movie_name,
        movies=[],
        genres=get_genres(),
    )


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    if movie_id >= len(df):
        return "Movie not found", 404

    movie  = df.iloc[movie_id]
    poster = build_poster_url(movie.get("poster_path", ""))
    _, recs = recommend(movie["original_title"])

    return render_template(
        "movie_detail.html",
        movie=movie,
        poster=poster,
        recommendations=recs,
    )

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)