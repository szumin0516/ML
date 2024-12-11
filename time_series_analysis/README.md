# Time series analysis - forecasting airline passengers

Time series data is a set of consecutive data points collected over time, which can be widely applied in various domains such as supply chain, business, and financial for predicting those future trends. Generally, the aim for such analysis is to identify future patterns or vairations based on historical data. In the proposed time series project, it is used to foresee the potential trend of airline passengers.

---

### Reqired packages
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
### Loading data
```python
df = pd.read_csv('AirPassengers.csv')
df.set_index('Month', inplace=True)
df.head()
```
### Visualization of historical data
```python
plt.figure(figsize=(20, 5))
plt.plot(df.index, df['#Passengers'], label='#Passengers')
plt.xlabel('Date')
plt.ylabel('#Passengers')
plt.title('#Passengers Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
```
### Visualization of seasonal decomposition
In order to look into current data more deeply, `seasonal_decompose` can be utilized to break down the time series dataset into 3 main components for further analyzing:
* **Trend:** It represents the long term trend for time series data.
* **Seasonal:** Seasonality indicates the repeating & short term fluctuations grouped by seasons or cycles.
* **Residual (noise):** It shows the random variability remain after dropping the **trend** and **seasonality**.
```python
# plot the components in the graph
sns.set(style='whitegrid')

plt.figure(figsize=(18,12))

# trend component
plt.subplot(411)
sns.lineplot(data=result.trend)
plt.title('Trend')
plt.xticks(rotation=90)

# seasonal component
plt.subplot(412)
sns.lineplot(data=result.seasonal)
plt.title('Seasonal')
plt.xticks(rotation=90)

# Residuals component
plt.subplot(413)
sns.lineplot(data=result.resid)
plt.title('Residuals')
plt.xticks(rotation=90)

# Original data
plt.subplot(414)
sns.lineplot(data=df['#Passengers'])
plt.title('Original Data')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
```
### Augmented Dickey-Fuller (ADF) test
ADF test is a statistical test that determines if a time series is stationary or not. 
* `AIC` is chosen in `autolag` argument for its advantage for balancing between model fits.
* `ADF statistic` indicates the stablization of the test result. The more negative the value is, the more stronger the evidence against the null hypothesis -> __the time series data is stationary.__
* `p value` demonstrates the siginificance of the test result. If p value < 0.5, reject the null hypothesis -> __the time series data is stationary.__
```python
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller Test

result = adfuller(df['#Passengers'], autolag='AIC') # Akaike Information Criterion
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
Then perform a **first-order difference** in the dataset. It refers to the differences between consecutive observations for make a time series stationary.
```python
# first order differencing
result = adfuller(df['#Passengers'].diff().dropna(), autolag='AIC')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
``` 
Also, **second-order difference** is calculated such as quadratic trends for previous difference (first-order difference). i.e, second-order difference is the first-order difference of the first-order difference.
```python
# second order differencing
result = adfuller(df['#Passengers'].diff().diff().dropna(), autolag='AIC')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
Visualize the original trend with its first-order difference and second-order difference altogether.
```python
# plot the differencing values
fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(df)
ax1.set_title('Original Time Series')
ax1.axes.xaxis.set_visible(False)

ax2.plot(df.diff())
ax2.set_title('1st Order Differencing')
ax2.axes.xaxis.set_visible(False)

ax3.plot(df.diff().diff())
ax3.set_title('2nd Order Differencing')
ax3.axes.xaxis.set_visible(False)

plt.show()
```
### Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
* **ACF:** It measures the correlation between a time series and its lagged versions over different time lags. It ranges from -1 to 1. The more postive the value is, the more strongly positivly correlated. Otherwise, the more negative the value is, the more strongly negativly correlated.
* **PACF:** It measures the correlation between a time series and its lagged versions while controlling for the influence of all shorter lags. It ranges from -1 to 1 as well.
```python
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
sm.graphics.tsa.plot_acf(df.diff().dropna(), lags=40, ax=ax[0])
sm.graphics.tsa.plot_pacf(df.diff().dropna(), lags=40, ax=ax[1])
plt.show()
```

### Model training
Seasonal Autoregressive Integrated Moving Average (SARIMA) model, the seasonal version of ARIMA model, is designed to predict data with seasonal patterns. Before applying SARIMA, defining some important parameters are required.
```python
p = 2 # pacf (autoregressive order)
d = 1 # first order difference (differencing order)
q = 1 # acf (moving average order)

P = 1 # seasonal autoregressive order
D = 0 # seasonal differencing order
Q = 3 # seasonal moving average order

# define the SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(df['#Passengers'], order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
fitted_model = model.fit()
print(fitted_model.summary())
```
### Prediction
Use the fitted SARIMA model to predict the number of passengers in the next 2 years.
The argument `periods` is to specify the number of periods to generate. `+1` ensures the result date can cover the whole forecast period including the start date.
`[1:]` is used to slice the date range to exclude the start date.
```python
# forecast for next 2 years
forecast_steps = 24
forecast = fitted_model.get_forecast(steps=forecast_steps)

# create the date range for the forecasted values
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='M')[1:].strftime('%Y-%m') # remove start date
```
and demonstrate the prediction result in DataFrame
```python
# create a forecast dataframe
forecast_df = pd.DataFrame({
    "Forecast": list(forecast.predicted_mean),
    "Lower CI": list(forecast.conf_int().iloc[:, 0]),
    "Upper CI": list(forecast.conf_int().iloc[:, 1])
}, index=forecast_index)

forecast_df.head()
```
Finally, viusalize the prediction result.
```python
# plot the forecast values

plt.figure(figsize=(12, 6))
plt.plot(df['#Passengers'], label='Historical Data')
plt.plot(forecast_df['Forecast'], label='Forecast Data')
plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='k', alpha=0.1)
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passengers Forecast')
plt.xticks(rotation=90)
plt.legend()
plt.show()
```