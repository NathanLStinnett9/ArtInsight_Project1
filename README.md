# ArtInsight_MachineLearning
**Machine Learning Modeling and Pipeline | Capstone Project 1**  
**Author:** Nathan Stinnett | **Date:** June 2026

---

## Overview

Developed a machine learning project analyzing 2,500 online art marketplace listings to predict artwork authenticity and recommend listing prices. Built scikit-learn classification and regression pipelines, incorporating exploratory data analysis, feature engineering, GridSearchCV hyperparameter tuning, and serialized model deployment.

- **Model 1 (Classification):** Predict whether a piece is an **Original** or a **Copy**
- **Model 2 (Regression):** Suggest a **data-driven listing price** based on artwork attributes

---

## Dataset

- **File:** `large_art_ecommerce_dataset.csv`
- **Source:** Kaggle — *"Art Market Dataset: Selling Paintings Prediction"* by jijagallery
- **Size:** 2,500 rows × 18 columns
- **Key features:** Style, Subject, Medium, Size, Mood, Price, Artist, Reproduction Type

---

## Project Structure
1. **Setup** — imports and configuration
2. **Load the Data** — shape, dtypes, missing value audit
3. **Exploratory Data Analysis** — price distribution, class balance, correlation analysis
4. **Feature Engineering & Preprocessing** — ColumnTransformer with OneHotEncoder + StandardScaler
5. **Model 1: Classification** — Logistic Regression, Random Forest, Gradient Boosting + GridSearchCV
6. **Model 2: Regression** — Ridge, Random Forest, Gradient Boosting + GridSearchCV
7. **Feature Importance** — top predictors visualized
8. **Business Summary** — findings, limitations, and recommendations

---

## Key Results

Both models performed close to their naive baselines after tuning. This is itself a meaningful finding:

- Listing attributes (style, subject, medium, size) are **not reliable predictors** of authenticity or price in this dataset
- The dataset captures **asking prices**, not verified sale/transaction prices
- Scraping hammer price and sold data either cost money for monthly subscription or was unsuccessful code to analyze page source code for needed info
- Only **11 distinct painters** across 2,500 rows limits artist-level signal

**Recommendation:** Richer data — artist sales history, real transaction outcomes, image-based scoring — is needed before a production model is viable.

---
