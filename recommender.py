import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="data/skincare.csv"):
    return pd.read_csv(path, encoding="latin-1")


def build_tfidf(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["description"])
    return tfidf_matrix


def recommend_products(skin_type, df, tfidf_matrix, top_n=10):
    # Multi-label safe matching
    matches = df[df["skintype"].str.contains(skin_type, case=False, na=False)]

    # Safe fallback
    if matches.empty:
        idx = 0
    else:
        idx = matches.index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[::-1][1:top_n + 1]
    return df.iloc[top_indices][["product_name", "brand"]]
