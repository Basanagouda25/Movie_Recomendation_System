# Movie Success Prediction & Recommendation System

This project involves a comprehensive end-to-end data science pipeline to analyze movie data, perform feature engineering, and build machine learning models to predict whether a movie will be a "Hit" based on its metadata.

## 🚀 Project Overview
The goal of this project is to process raw movie data (Genres, Cast, Director, etc.) and use advanced classification algorithms to predict movie success. A unique "Metadata Soup" was created to capture the essence of each film for improved model performance.

## 📊 Dataset
The dataset includes over 4,700 movies with features such as:
* **Numerical:** Budget, Revenue, Popularity, Runtime, and Vote Average.
* **Categorical/Text:** Movie Genre, Keywords, Tagline, Cast, and Director.

## 🛠️ Data Pipeline

### 1. Data Collection & Cleaning
* Handled missing values in text features (`Keywords`, `Tagline`, `Cast`, `Director`) by filling them with empty strings.
* Imputed missing `Runtime` values using the median.
* Dropped irrelevant columns like `Movie_Homepage`.

### 2. Feature Engineering
* **Metadata Soup:** Created a `combined_features` column by merging Genre, Keywords, Tagline, Cast, and Director into a single string for text vectorization.
* **Target Labeling:** Created a binary target `Is_Hit` (1 if Rating > 7, else 0).

### 3. Exploratory Data Analysis (EDA)
Performed 5 key visualizations:
* **Distribution of Ratings:** Understanding user voting patterns.
* **Top 5 Genre Popularity:** Using Box Plots to identify outliers and median popularity.
* **Budget vs. Revenue:** Scatter plot identifying financial correlations.
* **Top 10 Directors:** Identifying prolific creators in the dataset.
* **Correlation Heatmap:** Statistical analysis of numerical feature relationships.

## 🤖 Machine Learning Models
Three powerful classification models were trained and compared:
1. **Random Forest:** Used balanced class weights to handle dataset imbalance.
2. **Gradient Boosting:** Captured complex non-linear patterns.
3. **XGBoost:** The best performing model for identifying "Hits", optimized with `scale_pos_weight`.

## 📈 Evaluation Results

| Model | Accuracy | Recall (Hits) | AUC-ROC |
| :--- | :--- | :--- | :--- |
| Random Forest | 85% | 0.08 | 0.772 |
| Gradient Boosting | 84% | 0.09 | 0.779 |
| **XGBoost** | **75%** | **0.53** | **0.803** |

*Note: While Random Forest had higher overall accuracy, **XGBoost** was chosen as the superior model because it successfully identified 53% of successful movies (Hits), whereas other models mostly predicted the majority class (Flops).*

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Environment:** Kaggle Notebooks / Jupyter

## 📂 Project Structure
```bash
├── MovieRecommendationSystem.csv   # Raw Dataset
├── Movie_Recommendation.ipynb      # Complete Kaggle Notebook
└── README.md                       # Project Documentation
```
---

## 📝 Conclusion
By utilizing **TF-IDF Vectorization** on a combined metadata string, this project demonstrates that movie metadata (cast, director, and genre) contains significant predictive power regarding a film's critical reception.

Although the dataset is highly imbalanced, using **XGBoost with `scale_pos_weight`** allowed the model to maintain a strong **AUC of 0.803**, effectively distinguishing between *"Hits"* and *"Flops"* where traditional models struggled.

This indicates that while financial metrics are important, the creative **"metadata soup"** of a movie plays a crucial role in determining its success.

---

## ⭐ Future Improvements
- Hyperparameter tuning using GridSearchCV / Optuna  
- Deep learning models (LSTM / Transformers for text features)  
- Deployment using Flask / FastAPI  
- Building a real-time recommendation system  

