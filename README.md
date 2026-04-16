# Movie Success Prediction & Recommendation System

This project involves a comprehensive end-to-end data science pipeline to analyze movie data, perform feature engineering, and build machine learning models to predict whether a movie will be a "Hit" based on its metadata.

## 🚀 Project Overview
The goal of this project is to process raw movie data (Genres, Cast, Director, etc.) and use advanced classification algorithms to predict movie success. A unique "Metadata Soup" was created to capture the essence of each film for improved model performance.

## 📊 Dataset
The dataset includes over 4,700 movies from the [Movie Recommendation System Dataset](https://www.kaggle.com/datasets/ronadasakalesha/movierecommendationsystemdataset) with features such as:
* **Numerical:** Budget, Popularity, Runtime, and Vote Average.
* **Categorical/Text:** Movie Genre, Keywords, Tagline, Cast, and Director.

## 🛠️ Data Pipeline

### 1. Data Collection & Cleaning
* Handled missing values in text features (`Keywords`, `Tagline`, `Cast`, `Director`) by filling them with empty strings.
* Imputed missing `Runtime` values using the median.
* Dropped irrelevant columns like `Movie_Homepage`.

### 2. Feature Engineering
* **Metadata Soup:** Merged Genre, Keywords, Tagline, Cast, and Director into a single string for TF-IDF Vectorization.
* **Target Labeling:** Created a binary target `Is_Hit` (1 if Rating > 7, else 0).
* **Feature Selection:** Utilized `SelectFromModel` with a Random Forest estimator to identify the most impactful numerical features (Budget, Popularity, and Runtime).

### 3. Exploratory Data Analysis (EDA)
Performed 5 key visualizations:
* **Distribution of Ratings:** Visualizing user voting patterns.
* **Top 5 Genre Popularity:** Using Box Plots to identify outliers and median popularity.
* **Budget vs. Revenue:** Scatter plot identifying financial correlations.
* **Top 10 Directors:** Identifying prolific creators in the dataset.
* **Correlation Heatmap:** Statistical analysis of numerical feature relationships.

## 🤖 Machine Learning Models
Three powerful classification models were trained and compared:
1. **Random Forest:** Used balanced class weights to handle dataset imbalance.
2. **Gradient Boosting:** Captured complex non-linear patterns.
3. **XGBoost:** Optimized with `scale_pos_weight` to prioritize the identification of "Hits."

## 📈 Evaluation Results

| Model | Accuracy | Recall (Hits) | AUC-ROC |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **85.82%** | 0.17 | 0.772 |
| Gradient Boosting | 86.00% | 0.33 | 0.779 |
| **XGBoost** | 85.00% | **0.62** | **0.803** |

*Note: While Random Forest achieved high overall accuracy, **XGBoost** was chosen for its superior ability to correctly identify Hits (62% Recall), whereas the standard Random Forest was highly conservative.*

## 🔮 Interactive Prediction
The project includes a **Champion Model Selection** script that automatically picks the best-performing model to run real-time predictions. Users can input movie details to receive a success probability.

**Example Prediction (Interstellar):**
- **Input:** Budget: $165M, Director: Christopher Nolan, Popularity: 724
- **Result:** `HIT 🌟` (Confidence: 54.00%)

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Environment:** [Kaggle Notebooks](https://www.kaggle.com/code/basu25/movierecomendation)

## 📂 Project Structure
```bash
├── MovieRecommendationSystem.csv   # Raw Dataset
├── Movie_Recommendation.ipynb      # Complete Kaggle Notebook
└── README.md                       # Project Documentation
```
---

## 📝 Conclusion
By utilizing TF-IDF Vectorization on a combined metadata string, this project demonstrates that movie metadata (cast, director, and genre) contains significant predictive power regarding a film's critical reception.

Although the dataset is highly imbalanced, using XGBoost with scale_pos_weight allowed the model to maintain a strong ***AUC of 0.803**, effectively distinguishing between "Hits" and "Flops" where traditional models struggled. This indicates that while financial metrics are important, the creative "metadata soup" plays a crucial role in determining its success.

---

## ⭐ Future Improvements
- Hyperparameter tuning using GridSearchCV / Optuna  
- Deep learning models (LSTM / Transformers for text features)  
- Deployment using Flask / FastAPI  
- Building a real-time recommendation system  

