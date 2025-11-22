Movie Recommendation System

A Streamlit app that recommends movies similar to your favorite titles using Item-Based Collaborative Filtering (KNN).
This project demonstrates a full end-to-end Machine Learning workflow: data preprocessing, sparse matrix creation, model training, and real-time movie recommendations.

📚 Academic Project Info
Course: Data Science Technologies (CS4135)
Supervisor: Sir Muhammad Abdul Jabbar
Student: Abd-ul-ala Taha
Dataset: MovieLens Ratings Dataset
Model: Item-Based Collaborative Filtering (KNN)

🌐 [Open in Streamlit](https://cinemarecomendation.streamlit.app/)

⚠️ Note: If the app is not sleeping, it will open immediately. On first run, the pretrained models and datasets are automatically downloaded from Google Drive.

🌟 Features

Search for a movie and get top recommendations based on user ratings
Displays movie title, genres, and similarity scores
Configurable number of recommendations
Clean, responsive Streamlit UI
Auto-downloads required datasets and models from Google Drive
Academic project description included in-app

🛠 Technologies Used

Python 3.11
Streamlit
Pandas, NumPy
Scikit-learn (KNN)
Pickle / gdown for model loading
Sparse matrices for efficient computation

🚀 Run the App Locally

Clone the repository

git clone https://github.com/alachaudhary/MovieRecomendationEngine.git
cd MovieRecomendationEngine

Install dependencies

pip install -r requirements.txt

Run the Streamlit app

streamlit run streamlit_app.py

Open your browser
The app will be available at http://localhost:8501

📂 Project Structure
MovieRecomendationEngine/
│
├── models/                # Pretrained models and processed datasets (downloaded via Google Drive)
├── streamlit_app.py       # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md
└── ...


Note: Large model files are downloaded automatically from Google Drive on first run, so ensure internet access.

⚡ How It Works

The app loads the KNN model, user and movie mappings, and the sparse ratings matrix.

When you enter a movie title, it finds the closest matches based on user ratings.

Recommends the top N similar movies along with similarity scores.


📌 Notes

Ensure internet access for the first run to download models from Google Drive.

Requires compatible Python and NumPy versions to load pickled models.

✨ License

This project is licensed under the MIT License.
