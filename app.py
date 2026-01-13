import streamlit as st
from recommender import load_data, build_tfidf, recommend_products

st.set_page_config(
    page_title="Skin Care Tonic Recommender",
    layout="centered"
)

st.title("🧴 Skin Care Tonic Recommendation System")
st.write("Get personalized skincare product recommendations based on your skin type.")

@st.cache_data
def load_all():
    df = load_data()
    tfidf_matrix = build_tfidf(df)
    return df, tfidf_matrix

try:
    df, tfidf_matrix = load_all()

    # extract unique primary skin types safely
    skin_types = sorted(
        set(s.split(",")[0].strip() for s in df["skintype"])
    )

    skin = st.selectbox("Select your skin type", skin_types)

    if st.button("Get Recommendations"):
        recs = recommend_products(skin, df, tfidf_matrix)
        st.subheader("Top Recommended Products")
        st.dataframe(recs)

except Exception as e:
    st.error("❌ Something went wrong while running the app.")
    st.exception(e)
