# Flight-Fare-Predictor

A machine-learning / data-science project to predict flight ticket fares based on historical flight data and key features (such as airlines, route, date, stops, departure/arrival times).

## 🚀 Overview

This repository implements a workflow to:

* Ingest flight booking or fare dataset (with flight details + price)
* Perform preprocessing (cleaning, feature engineering)
* Train predictive models (regression) to estimate flight fare given input features
* Evaluate the accuracy of predictions (errors, R², etc)
* Optionally build a simple interface or API to query predicted fare for new inputs

## 🗂 Repository Structure

Here’s a suggested (and adjust according to your actual project) layout:

```
flight-fare-predictor/
│
├── data/                     # raw and/or processed datasets  
│   ├── raw/                  # original CSV/Excel files  
│   └── processed/            # cleaned & feature-engineered data  
├── notebooks/                # Jupyter notebooks for EDA, modelling  
│   ├── 01_EDA.ipynb  
│   ├── 02_FeatureEngineering.ipynb  
│   └── 03_ModelTraining.ipynb  
├── src/                      # source code / modules  
│   ├── data_preprocess.py  
│   ├── feature_engineer.py  
│   ├── model_train.py  
│   └── model_predict.py  
├── models/                   # saved trained model(s) (pickle, joblib, etc)  
├── reports/                  # output reports, evaluation metrics, plots  
├── requirements.txt          # Python dependencies  
├── README.md                 # this file  
└── LICENSE                   # Licence file  
```

*(Modify folder names/paths according to your actual repo.)*

## 🛠 Features

* Loads historic flight fare data and relevant features (airline, source, destination, stops, departure time, arrival time, date, etc)
* Data cleaning: handling missing values, converting dates, encoding categories (airline, routes)
* Feature engineering: extracting day/month/year, duration, number of stops, departure hour, etc
* Model training: regression models (e.g., Linear Regression, Random Forest Regressor, XGBoost) to predict fare
* Model evaluation using metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), R² (Coefficient of Determination)
* (Optionally) Saving the trained model and using a `predict` module/script to predict fare for new inputs
* (Optionally) Visualisations: distribution of fares, feature importance, error plots

## 📦 Prerequisites & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Taher284/flight-fare-predictor.git
   cd flight-fare-predictor
   ```
2. (Recommended) Create a Python virtual environment:

   ```bash
   python3 -m venv venv  
   source venv/bin/activate     # (Windows: venv\Scripts\activate)  
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Prepare data:

   * Place your flight fare dataset(s) in `data/raw/`
   * Update any path or config in `data_preprocess.py` or config file
5. Run preprocessing and training:

   * For example:

     ```bash
     python src/data_preprocess.py  
     python src/model_train.py  
     ```
6. Predict for new input:

   ```bash
   python src/model_predict.py --input "airline=…, source=…, destination=…, stops=…, date=…"  
   ```

   *(Adjust parameters according to your script.)*

## 📊 Workflow

Typical pipeline:

* **Load & Clean Data**: Read CSV/Excel, drop unused columns, handle missing data, convert date/time to usable format
* **EDA (Exploratory Data Analysis)**: Analyse fare distributions, correlation with stops, route, departure time, etc
* **Feature Engineering**: Derive new features (hour of departure, day of week, month, duration, number of stops)
* **Split Data**: Into training and validation/test sets (ensure randomness or time-based split if appropriate)
* **Train Model(s)**: Fit chosen regression model(s) on training set, tune hyper-parameters
* **Evaluate Model**: Report metrics (MAE, MSE, R²), plot prediction vs actual, inspect residuals
* **Save Model & Predictions**: Persist model for reuse, optional script/interface for predictions
* **(Optional) Deployment**: Wrap model in a simple web-app or API endpoint to allow user input and prediction in real-time

## ✅ Usage & Examples

* Example command to train model:

  ```bash
  python src/model_train.py --epochs 100 --model_type random_forest
  ```
* Example prediction:

  ```bash
  python src/model_predict.py --airline "IndiGo" --source "Delhi" --destination "Mumbai" --stops 0 --date "2025-12-15" --departure_time "09:30"  
  ```
* Review `reports/` folder for summary of model performance, plots of feature importance and errors.

## 📈 Evaluation Metrics

Since this is a regression problem (fare prediction), key metrics include:

* **Mean Absolute Error (MAE)**: average absolute error between predicted fare and actual fare
* **Mean Squared Error (MSE)**: average squared error (penalises larger errors more)
* **Root Mean Squared Error (RMSE)**: square-root of MSE (same units as fare)
* **R² Score (Coefficient of Determination)**: how much variance in the fare the model explains
* **Residual analysis**: plot actual vs predicted, error distribution, check for bias (under/over-prediction)

## 🧩 Extending the Project

* Add more advanced models: Gradient Boosting (XGBoost, LightGBM), Neural Networks
* Incorporate time-series/temporal aspects: fare change as function of booking lead-time, seasonal trends
* Feature enrichment: include external data such as holidays/seasonality, competitor pricing, airline promotions
* Build a UI/web-app: allow users to input flight details and get fare prediction live
* Automate pipeline: scheduling new data ingestion, model retraining, and deployment
* Improve interpretability: provide feature contributions using SHAP or LIME
* Deploy in cloud: wrap model as REST API (Flask/FastAPI) and containerise (Docker)
* Add logging and error-handling for production readiness

## 👥 Contributing

Contributions are welcome! If you’d like to contribute:

1. Fork the repository
2. Create a branch for your feature or fix:

   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes with clear message
4. Update documentation and tests (if applicable)
5. Push your branch and open a Pull Request

