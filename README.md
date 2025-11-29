Project Overview

Life expectancy is influenced by a wide variety of socio-economic and health-related factors. This project explores these relationships and builds a predictive model using different machine-learning algorithms.

The following models were trained and evaluated:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor (Best Performing Model)

Feature selection was performed using Lasso, and hyperparameters were optimized using RandomizedSearchCV.


Project Structure

.
├── data/
│   └── life_expectancy.csv
├── notebooks/
│   └── exploration_and_modeling.ipynb
├── models/
│   └── xgboost_model.pkl
├── app/
│   └── streamlit_app.py
├── README.md
└── requirements.txt


Technologies & Libraries

Python

Pandas, NumPy

Scikit-Learn

XGBoost

Matplotlib / Seaborn

Streamlit

RandomizedSearchCV

Lasso Regression (Feature Selection)

1. Install Dependencies
pip install -r requirements.txt

2. Run the Streamlit App
streamlit run app/streamlit_app.py

<img width="1062" height="740" alt="Image" src="https://github.com/user-attachments/assets/7e1afe55-d822-400b-8431-e95285ac8b6f" />

