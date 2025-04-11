# Baseball Hit Outcome Prediction Project ⚾

## Overview
This project aims to predict the outcome of a batted ball in baseball (e.g., Single, Double, Home Run, Out) based on quantifiable pitch and swing metrics sourced from Statcast data via Baseball Savant. The goal is to leverage machine learning to provide insights for players, coaches, and analysts, potentially identifying key factors that lead to successful (or unsuccessful) batted ball events.

This end-to-end project demonstrates skills in web scraping, data cleaning, feature engineering, model training/evaluation, hyperparameter optimization, model interpretation, cloud integration, and interactive dashboard creation.

## 📊 Project Overview

### 🔁 Workflow Steps

1. **Data Acquisition**  
   Scrapes Statcast data from [Baseball Savant](https://baseballsavant.mlb.com) using BeautifulSoup and stores raw CSV files locally or in Google Cloud Storage (GCS).

2. **Data Preprocessing**  
   - Cleans missing or noisy data.
   - Standardizes and encodes features.
   - Engineers domain-specific features: swing efficiency, pitch location categorization, pitch type categorization, outcome grouping

3. **Model Training**  
   Trains four types of classifiers:
   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost  
   The best model is selected based on accuracy.

4. **Hyperparameter Optimization**  
   Applies Bayesian optimization (via `scikit-optimize`) to fine-tune model performance.

5. **Feature Importance Analysis**  
   Combines:
   - Built-in feature importance
   - Permutation importance
   - SHAP values  
   Results are saved as CSV and plots under `reports/feature_importance`.

6. **Model Deployment & Storage**  
   - Saves models locally and uploads the latest versions to GCS.
   - Supports weekly automated data refreshes via GCP Cloud Functions and Scheduler.

7. **📊 Streamlit Dashboard**  
   _[To Be Added]_ – A Streamlit dashboard will allow users to:
   - Upload swing/pitch inputs and view predicted outcomes.
   - Toggle predictions by player/team.
   - Visualize feature importances and model confidence.

---

## 📁 Directory Structure

```
ml-baseball-outcome-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── latest/
│   ├── final/
│   ├── hyperparams/
│   ├── optimized/
├── reports/
│   ├── feature_importance/
│   └── final_model/
├── logs/
├── src/
│   ├── data/
│   ├── models/
│   └── cloud/
```

---

## 🧠 Final Model Summary (may change every model run)

- **Model Type**: `CatBoostClassifier`
- **Trained On**: 2025-04-10
- **Accuracy**: **41.5%**
- **Features Used (13)**:
  - `bat_speed`, `swing_length`, `pz`, `px`, `horizontal_release` `spin_rate`, `swing_efficiency_ratio`, `arm_angle`, `speed_differential`, `vertical_release`, `extension`, `perceived_velocity`, `pitch_velocity`
- **Hyperparameters**:
```json
{
  "iterations": 472,
  "learning_rate": 0.010,
  "depth": 10,
  "l2_leaf_reg": 6.56,
  "bootstrap_type": "Bernoulli",
  "grow_policy": "SymmetricTree"
}
```

---

## 🚀 Google Cloud Integration

- Deployable via:
  - `cloud_function.py`
  - `deploy_cloud_function.sh`
- Weekly scheduled data scraping
- Data stored to:
  - `gs://baseball-ml-data/raw/`
  - `gs://baseball-ml-data/latest/`

---

## 📈 Future Improvements

- **Per-Player Model Training**  
  Train a separate model per player to allow personalized predictions. This will enable the Streamlit dashboard to offer a toggle per player/team and return model predictions based on their unique swing/pitch style.

---

## 🌐 Streamlit Dashboard

👉 **[Will insert dashboard link here once deployed]**

---

## 🛠️ Setup Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run scraping
python src/data/scrape_statcast.py

# Run preprocessing
python src/data/process_data.py

# Run training
python src/models/train_model.py

# Optimize hyperparameters
python src/models/bayesian_optimization.py

# Perform feature analysis
python src/reports/feature_analysis.py

# Train final model
python src/models/final_model_training.py
```

---

## 🧾 Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
shap
xgboost
lightgbm
catboost
mlflow
joblib
beautifulsoup4
requests
google-cloud-storage
functions-framework
```
