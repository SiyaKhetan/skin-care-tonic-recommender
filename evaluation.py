print("EVALUATION STARTED")

from recommender import load_data, build_tfidf, recommend_products

df = load_data()
print("DATA LOADED:", df.shape)

tfidf_matrix = build_tfidf(df)
print("TFIDF BUILT")


def precision_at_k_multilabel(recs_df, full_df, skin_type, k=5):
    merged = recs_df.merge(
        full_df[["product_name", "skintype"]],
        on="product_name",
        how="left"
    )

    top_k = merged.head(k)

    matches = top_k["skintype"].apply(
        lambda x: skin_type.lower() in x.lower()
    )

    return matches.sum() / k


scores = []

print("SKIN TYPES:", df["skintype"].unique())

for skin in df["skintype"].unique():
    primary_skin = skin.split(",")[0].strip()
    print("Evaluating:", primary_skin)

    recs = recommend_products(primary_skin, df, tfidf_matrix)
    score = precision_at_k_multilabel(recs, df, primary_skin)

    print(primary_skin, "Precision@5:", round(score, 2))
    scores.append(score)

print("DONE")
print("Average Precision@5:", round(sum(scores) / len(scores), 2))
