# 🧴 Skin Care Tonic Recommendation System
A content-based machine learning recommendation system that suggests suitable skincare tonic products based on a user’s skin type.  
The system is built using TF-IDF vectorization and cosine similarity and deployed as a live Streamlit web application.



## 🌐 Live Demo
👉 https://skin-care-tonic-recommender-5at2dcvey6ka9yq9nnmqk5.streamlit.app/


## 📌 Problem Statement
Choosing the right skincare product is difficult due to the wide variety of products and different skin types.  
This project aims to recommend the most relevant skincare tonics by analyzing product descriptions and matching them with the user’s skin type.


## 🧠 Approach
- Performed data preprocessing on a dataset of 1,210 skincare products
- Used TF-IDF Vectorization to convert product descriptions into numerical vectors
- Applied Cosine Similarity to measure similarity between products
- Implemented a content-based recommendation engine
- Handled multi-label skin types (e.g., Normal, Dry, Oily)
- Built an interactive UI using Streamlit
- Deployed the application on Streamlit Cloud



## 📊 Model Evaluation
- Evaluated recommendations using Precision@5
- Used multi-label relevance evaluation (skin type containment instead of exact match)
- Achieved an average Precision@5 of ~48%, which is realistic for content-based recommenders without user interaction data


## 🛠 Tech Stack
- Programming Language: Python  
- Libraries: pandas, numpy, scikit-learn  
- ML Techniques: TF-IDF, Cosine Similarity  
- Web Framework: Streamlit  
- Deployment: Streamlit Cloud  
- Version Control: Git & GitHub  


## 📂 Project Structure
skin-care-tonic-recommender/
├── app.py # Streamlit application
├── recommender.py # Recommendation logic
├── evaluation.py # Model evaluation
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── data/
└── skincare.csv # Dataset

# #▶️ How to Run Locally
```bash
git clone https://github.com/SiyaKhetan/skin-care-tonic-recommender.git
cd skin-care-tonic-recommender
pip install -r requirements.txt
streamlit run app.py


##🚀 Future Improvements
Improve ranking with hybrid models (price, brand, ratings)
Enhance UI and add product images

👩‍💻 Author

Siya Khetan
3rd Year B.Tech Student





