# 🌦️ Weather Prediction using Time Series (Helsinki 2015–2019)

This project focuses on predicting daily average temperatures using time series forecasting techniques. The data is collected from the weather station 2978 in Helsinki, Finland, spanning from January 2015 to September 2019.

---

## 📊 Data Source

The dataset was originally obtained from [rp5.ru](http://rp5.ru/), containing daily average temperature data. In this notebook, we analyze and forecast temperature values using time series modeling and evaluate prediction accuracy.

---

## 📁 File Overview

- `Weather_prediction_timeseries.ipynb`: Jupyter notebook containing the full code for data preprocessing, modeling, forecasting, and evaluation.
- `README.md`: Project description and setup guide.

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Statsmodels

---

## 📈 Methodology

1. **Data Preprocessing**  
   - Handling missing values  
   - Date parsing  
   - Visual exploration of temperature trends  

2. **Forecasting**  
   - One-step-ahead predictions using time series models  
   - Evaluation using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

3. **Error Metrics**
   - **MSE**:  
     \[
     \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
     \]
   - **RMSE**:  
     \[
     \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
     \]

---

## 🔍 Sample Code

```python
from sklearn.metrics import mean_squared_error

# Calculate RMSE
rmse = mean_squared_error(y_truth, y_predicted, squared=False)

# Or manually:
import numpy as np
rmse = np.sqrt(((y_truth - y_predicted) ** 2).mean())
