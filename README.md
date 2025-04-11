# Baseball Hit Outcome Prediction Project ⚾

This project aims to predict the outcome of a batted ball in baseball (e.g., Single, Double, Home Run, Out) based on quantifiable pitch and swing metrics sourced from Statcast data via Baseball Savant. The goal is to leverage machine learning to provide insights for players, coaches, and analysts, potentially identifying key factors that lead to successful (or unsuccessful) batted ball events.

This end-to-end project demonstrates skills in web scraping, data cleaning, feature engineering, model training/evaluation, hyperparameter optimization, model interpretation, cloud integration, and interactive dashboard creation.

---

## 🌐 Streamlit Dashboard

👉 **[Will insert dashboard link here once deployed]**

---

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
   - Tracked all experiments and models using **MLflow**, including:
     - Model parameters, metrics, artifacts, and versioning
     - Comparison across multiple classifier types and optimization stages

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
   _[To Be Added]_ – A Streamlit dashboard will allow users to upload swing/pitch inputs and view predicted outcomes.

---

## 🧠 Model Summary (may change every model run)

### Model Performance
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
The final model achieved an accuracy of 41.51% on the test set. While this might not seem high in absolute terms, it's important to consider the context of baseball hit outcomes prediction, which is inherently challenging due to the number of variables involved and the natural variability in baseball outcomes.

Our model performs significantly better than random guessing (25% for a 4-class problem) and provides valuable insights, particularly for singles prediction where it achieves a recall of 75.7%. The model struggles more with extra-base hits, which is expected given their relative rarity and the subtle differences in swing mechanics that distinguish them from singles and home runs.

These results provide a solid foundation for predicting hit outcomes and can be particularly useful for strategic decision-making when combined with player-specific analysis. The model's strength in detecting singles makes it especially valuable for defensive positioning and pitching strategy.

### Feature Importance and Insights
**Key Predictive Features**
The most important features for predicting hit outcomes, based on both feature importance and permutation importance are:
1. Bat Speed: The most critical factor in determining hit outcomes
2. Swing Length: Shorter, more compact swings correlate with singles, while longer swings associate with extra-base hits and home runs
3. Pitch Vertical Location (pz): Significantly impacts the type of contact made
4. Pitch Horizontal Location (px): Affects the batter's ability to make solid contact
5. Horizontal Release Point: Impacts the batter's ability to pick up the pitch
6. Spin Rate: Higher spin rates create more movement, affecting contact quality
7. Swing Efficiency Ratio: Measure of how efficiently the batter converts swing length to bat speed
8. Arm Angle: Affects pitch perception and timing
9. Speed Differential: Difference between pitch velocity and bat speed
10. Vertical Release Point: Affects the batter's perception of the pitch

**Statistical Insights by Outcome Type**
Analysis of the feature distributions by outcome type reveals clear patterns:
- Home Runs: Associated with the highest bat speed (mean: 75.2 mph) and longest swing lengths (mean: 7.52 ft)
- Extra-Base Hits: Show moderately high bat speed (mean: 73.5 mph) with slightly shorter swings
- Singles: Feature lower bat speeds (mean: 70.9 mph) with the shortest average swing length (mean: 7.16 ft)
- Outs: Characterized by high negative speed differentials (mean: -17.3), indicating a larger gap between pitch speed and bat speed

The speed differential (bat speed minus pitch velocity) shows a clear trend across outcomes:
- Home Runs: -14.01 mph (smallest difference)
- Extra-Base Hits: -15.71 mph
- Outs: -17.31 mph
- Singles: -18.42 mph (largest difference)

This suggests that minimizing the speed gap between bat and pitch is critical for power hitting.

---

## 📁 Directory Structure

```
ml-baseball-outcome-predictor/
├── data/
│   ├── raw/            # Raw data scraped from Baseball Savant
│   └── processed/      # Cleaned and processed data
├── src/
│   ├── data/           # Data collection and processing scripts
│   │   ├── scraper.py  # Web scraping functionality
│   │   └── processor.py # Data preprocessing
│   ├── models/         # Model training and evaluation
│   │   ├── train_model.py          # Model training
│   │   ├── bayesian_optimization.py # Hyperparameter tuning
│   │   ├── train_final_model.py    # Final model training
│   │   └── feature_analysis.py     # Feature importance analysis
├── models/             # Saved models and artifacts
├── cloud/              # Cloud deployment resources
│   ├── cloud_function.py           # GCP Cloud Function
│   └── deploy_cloud_function.sh    # Deployment script
├── app.py              # Streamlit dashboard application
└── README.md           # Project documentation
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

## 🛠️ Setup Instructions
### Prerequisites
- Python 3.9+
- Google Cloud account (for cloud deployment features)
- MLB Statcast access

```bash
# Install dependencies
pip install -r requirements.txt

# Run scraping
python src/data/scraper.py --seasons 2023 2024 2025

# Run preprocessing
python src/data/processor.py

# Run training
python src/models/train_model.py

# Optimize hyperparameters
python src/models/bayesian_optimization.py

# Perform feature analysis
python src/reports/feature_analysis.py

# Train final model
python src/models/train_final_model.py

# Deploy cloud function
cd src/cloud
./deploy_cloud_function.sh your-project-id us-central1

# Run Streamlit
streamlit run app.py
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
streamlit
google-cloud-storage
functions-framework
```

---

## 📈 Future Improvements

- **Model Training Automation**  
  Automate model training and evaluation to trigger after each new data scraping job using Cloud Scheduler and Cloud Functions. This would ensure the model stays up to date without manual intervention.
  - Note: Due to limited GCP credits, model retraining is currently performed manually to control compute usage.
- **Per-Player Model Training**  
  Train a separate model per player to allow personalized predictions. This will enable the Streamlit dashboard to offer a toggle per player/team and return model predictions based on their unique swing/pitch style.
- **Pitcher/Batter Handedness**  
  Train a model that considers pitcher and batter handedness and apply that has a mutable value within the dashboard.

---

## 💡 Acknowledgments
- MLB Statcast for providing the data
- The baseball analytics community for research and insights
- Open source ML and data science libraries
