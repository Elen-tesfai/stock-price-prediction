# 📈 stock-price-prediction

Predicting future stock prices using machine learning models and historical S&P 500 stock data.

---

## 🔍 Overview

This project explores machine learning approaches to forecast the next day's closing price for S&P 500 stocks. The notebook includes full preprocessing, feature engineering, model training, evaluation, and future forecasting.

- **Model Used:** Random Forest Regressor  
- **Dataset:** [S&P 500 Stock Data from Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)  
- **Goal:** Predict next-day closing price  
- **Performance Metric:** RMSE (Root Mean Squared Error)

---

## 🛠 Features and Workflow

1. **Data Loading & Cleaning** – Removed missing values, filtered relevant stock data (e.g., AAPL).
2. **Feature Engineering** – Added lag features, 7-day and 30-day moving averages.
3. **Model Training** – Trained Linear Regression, Support Vector Regression, Random Forest, and Gradient Boosting models.
4. **Hyperparameter Tuning** – Used GridSearchCV to tune Random Forest.
5. **Evaluation** – Used RMSE, MAE, R² metrics to compare model accuracy.
6. **Forecasting** – Produced 7-day forecasts and evaluated predictions with real data.
7. **Visualization** – Saved performance and forecast charts to `images/` folder.
8. **Documentation** – Modular, clean code inside Jupyter Notebook with clear step-by-step explanations.

---

## ✅ Final Results

- **Best Model:** Random Forest  
- **Best Parameters:** `max_depth=3`, `min_samples_split=10`, `n_estimators=10`  
- **Test RMSE:** `11.1352`  
- **Forecast RMSE (7-day):** `3.5490`

---

## 🖼 Key Visualizations

- 📉 AAPL Closing Prices Over 5 Years  
  `images/AAPL_closing_Over_5years_prices.png`

- 📊 Predicted vs Actual Close Prices  
  `images/predicted_vs_actual.png`

- 🪄 Feature Importance from Random Forest  
  `images/feature_importance.png`

- 🔮 7-Day Price Forecast  
  `images/7Days_forecast.png`

- 📈 Forecast Evaluation  
  `images/forecast_eval.png`

---
## 📓 Running the Notebook

Open and run the notebook `stock_forecasting.ipynb` to explore the data, model training, evaluation, and forecasting process step by step.

## 📁 Project Structure

stock-price-prediction/
│
├── data/
│   └── all_stocks_5yr.csv
├── images/
│   ├── AAPL_closing_Over_5years_prices.png
│   ├── predicted_vs_actual.png
│   ├── feature_importance.png
│   ├── 7Days_forecast.png
│   └── forecast_eval.png
├── stock_forecasting.ipynb
├── .gitignore   
├── README.md
└── requirements.txt

---

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install using:

```bash
pip install -r requirements.txt
```
# 📚 References

- S&P 500 Dataset on [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- Tesfai, E. (2025). *S&P 500 Stock Price Prediction Project*. Retrieved from [https://github.com/Elen-tesfai/stock-price-prediction](https://github.com/Elen-tesfai/stock-price-prediction)

---

## 🧠 Author

**Elen Resfai**  
📍 Data Science enthusiast passionate about forecasting and applied machine learning.