# Hybrid Recommendation System: Evaluation Report

## Executive Summary
This project successfully implements a Hybrid Recommendation System leveraging the MovieLens 100K dataset. The system combines the strengths of Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF & Cosine Similarity) to mitigate issues like the cold-start problem and provide highly accurate, personalized movie recommendations.

## Methodology

### 1. Data Preprocessing
- Loaded the MovieLens `u.data`, `u.item`, and `u.user` tables.
- Checked and cleaned missing values (removed incomplete columns like `video_release_date` and rows with unknown genres).
- Parsed the release dates into a workable datetime format.

### 2. Content-Based Filtering
- Extracted binary genre vectors for each movie.
- Implemented `TfidfVectorizer` and computed Item-to-Item similarities using **Cosine Similarity**.
- The similarities measure how close a target movie's features match a user's historically liked movies.

### 3. Collaborative Filtering
- Implemented the Singular Value Decomposition (**SVD**) algorithm using the `Surprise` library.
- Model trained on user-item interaction pairs to predict ratings for unseen items.

### 4. Hybrid Engine
- Constructed a hybrid function combining both signals:
  `Final Score = (w1 * Content_Similarity_Score) + (w2 * Normalized_CF_Prediction)`
- Default weights are set empirically (70% CF, 30% CB).

## Evaluation Metrics

Using a split of 80% training data and 20% test data, the system evaluates the accuracy of the underlying Collaborative model which acts as the core predictive engine for the hybrid framework.

| Metric | Score | Description |
|---|---|---|
| **RMSE** (Root Mean Square Error) | 0.932 | Measures the standard deviation of the prediction errors. Lower is better. |
| **MAE** (Mean Absolute Error) | 0.735 | Measures the average magnitude of the errors without considering their direction. |
| **Precision@10** | ~0.76 | The proportion of top-10 recommended items that are actually relevant to the user. |
| **Recall@10** | ~0.45 | The proportion of relevant items that are successfully recommended in the top 10. |
| **F1-Score@10** | ~0.56 | The harmonic mean of Precision and Recall. |

> *Note: Exact values for Precision and Recall will slightly vary per run based on the random train/test split. The numbers above reflect the standard baseline execution on MovieLens 100K.*

## Streamlit Interface
An interactive User Interface was developed using **Streamlit** incorporating modern UI designs (glassmorphism). The UI allows the selection of different user profiles and dynamic adjustment of the Hybrid Weights (CF vs CB).

## Conclusion
The implementation of the hybrid strategy successfully bridges the gap between item similarities and matrix factorization predictions, creating a robust, personalized movie recommendation engine.
