```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
```


```python
# Fetch data from Yahoo Finance
crude_oil = yf.Ticker("CL=F")
df = pd.DataFrame(crude_oil.history(start="2023-10-31", end="2024-10-31"))

df = df.drop(columns=['Dividends','Stock Splits'])
```


```python

# Volume Analysis
# ---------------------------
# Basic statistics on volume
average_vol = df['Volume'].mean()
max_vol = df['Volume'].max()
max_date = df['Volume'].idxmax()
vol_today = df['Volume'].iloc[-1:]

# Volume percentage comparisons
today_pct_avg = (vol_today / average_vol) * 100
today_pct_max = (vol_today / max_vol) * 100

# Rolling average of volume
roll_avg = df['Volume'].rolling(window=5).mean()
df['roll_avg_5days'] = roll_avg

# Volume threshold calculations
vol_std = df['Volume'].std()
high_vol_threshold = average_vol + (2 * vol_std)
high_vol_day = df[df['Volume'] > high_vol_threshold]

# 20-day moving average of volume for trend analysis
vol_20ma = df['Volume'].rolling(window=20).mean()
df['roll_avg_20days'] = vol_20ma
vol_last5 = vol_20ma[-5:].mean()
vol_last25 = vol_20ma[-25:-20].mean()
trend = "Increasing" if vol_last5 > vol_last25 else "Decreasing"

# Volume category labeling
df['Volume_Category'] = 'Normal'
df.loc[df['Volume'] > high_vol_threshold, 'Volume_Category'] = 'High'

trend

```




    'Decreasing'




```python
summary_analysis = {
    'average_volume': average_vol,
    'max_volume':  max_vol,
    'max_date': max_date,
    'volume_today': vol_today,
    'pct_today_vs_avg': today_pct_avg,
    'pct_today_vs_max': today_pct_max,
    'volume_std': vol_std,
    'high_vol_threshold': high_vol_threshold
}
summary_analysis
```




    {'average_volume': 315253.03571428574,
     'max_volume': 668216,
     'max_date': Timestamp('2024-10-01 00:00:00-0400', tz='America/New_York'),
     'volume_today': Date
     2024-10-30 00:00:00-04:00    284795
     Name: Volume, dtype: int64,
     'pct_today_vs_avg': Date
     2024-10-30 00:00:00-04:00    90.338543
     Name: Volume, dtype: float64,
     'pct_today_vs_max': Date
     2024-10-30 00:00:00-04:00    42.620201
     Name: Volume, dtype: float64,
     'volume_std': 97726.66741528806,
     'high_vol_threshold': 510706.3705448619}




```python
# Return Calculations
# ---------------------------
# Daily and log returns
daily_return = df['Close'].pct_change() * 100
log_return = np.log(df['Close'] / df['Close'].shift(1)) * 100

# Add returns to the DataFrame
df['daily_return'] = daily_return
df['log_return'] = log_return

# Correlation between volume and returns
correlation = df['Volume'].corr(df['daily_return'])

# Aggregated analysis of volume categories
volume_return_analysis = df.groupby('Volume_Category').agg({
    'daily_return': ['mean', 'std', 'count'],
    'Volume': 'mean'
})

volume_return_analysis
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">daily_return</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>Volume_Category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>High</th>
      <td>-1.761476</td>
      <td>2.925980</td>
      <td>4</td>
      <td>587984.500000</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>-0.019917</td>
      <td>1.901558</td>
      <td>247</td>
      <td>310854.141129</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Volatility Calculations
# ---------------------------
# Rolling volatility (20-day)
daily_volatility = df['daily_return'].rolling(window=20).std()
df['daily_volatility'] = daily_volatility

# Annualized volatility
yearly_volatility = df['daily_return'].std() * np.sqrt(252)

# Daily return statistics
avg_daily_return = daily_return.mean()
std_daily_return = daily_return.std()

# Volume on Up and Down Days
# ---------------------------
up_day = df[df['daily_return'] > 0]
down_day = df[df['daily_return'] < 0]




up_day_avg = up_day['Volume'].mean()
down_day_avg = down_day['Volume'].mean()

print([yearly_volatility, avg_daily_return, std_daily_return, up_day_avg,down_day_avg])
```

    [30.57051874046998, -0.04767065027715895, 1.9257616675600733, 299393.425, 330058.03846153844]



```python
# Price Range Analysis
# ---------------------------
# Daily price range and percentage range
price_range = df['High'] - df['Low']
df['price_range'] = price_range
daily_range_pct = (price_range / df['Open']) * 100
df['daily_range_pct'] = daily_range_pct

# Sorting by price range and statistics
price_range_sort = df.sort_values('price_range')
avg_price_range = price_range.mean()
std_price_range = price_range.std()

# Large price range days
large_range = df[df['price_range'] > (2 * std_price_range)]

# Monthly average price range
monthly_range = df.groupby(df.index.month)['price_range'].mean()

# Seasonality in price range
df['month'] = df.index.month
winter_month = df[df['month'].isin([12, 1, 2])]
summer_month = df[df['month'].isin([6, 7, 8])]

winter_avg = winter_month['price_range'].mean()
summer_avg = summer_month['price_range'].mean()

df['price_range'].plot()
```




    <Axes: xlabel='Date'>




    
![png](output_6_1.png)
    



```python
# Pivot Points and Support/Resistance Levels
# ---------------------------
df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
df['first_res'] = (2 * df['pivot_point']) - df['Low']
df['first_sup'] = (2 * df['pivot_point']) - df['High']
df['second_res'] = df['pivot_point'] + (df['High'] - df['Low'])
df['second_sup'] = df['pivot_point'] - (df['High'] - df['Low'])

# Plot pivot points
#df[['first_res', 'first_sup']].plot()
#df[['second_res', 'second_sup']].plot()
df[['first_res', 'first_sup', 'second_res', 'second_sup']].plot()
```




    <Axes: xlabel='Date'>




    
![png](output_7_1.png)
    



```python
# Local Highs and Lows (20-day rolling window)
# ---------------------------
window = 20
df['local_high'] = df['High'].rolling(window=window, center=True).max()
df['local_low'] = df['Low'].rolling(window=window, center=True).min()
```


```python
# Cluster Analysis by Price Bins
# ---------------------------
price_bins = pd.qcut(df['Close'], q=10)
cluster_analysis = df.groupby(price_bins)['Volume'].sum()
plt.boxplot(cluster_analysis)
print(cluster_analysis)
```

    Close
    (65.749, 70.371]    9795964
    (70.371, 72.248]    6783275
    (72.248, 73.823]    8843928
    (73.823, 75.554]    7823457
    (75.554, 77.045]    7929336
    (77.045, 78.016]    8469383
    (78.016, 78.997]    8021229
    (78.997, 80.996]    6948550
    (80.996, 82.802]    6980015
    (82.802, 86.91]     7848628
    Name: Volume, dtype: int64


    /var/folders/yv/5mf1zhqj65964f9nd_jlwvlh0000gn/T/ipykernel_3504/1584836889.py:4: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      cluster_analysis = df.groupby(price_bins)['Volume'].sum()



    
![png](output_9_2.png)
    



```python
# Gap Analysis 
# ---------------------------

# Calculating gap size
df['Gap'] = df['Open']-df['Close'].shift(1)
df['Gap_size'] = df['Gap'].abs()
gap_size_avg = df['Gap_size'].mean()
gap_size_std = df['Gap_size'].std()

# Filled gaps
df['Gap_Filled'] = False
for i in range(1, len(df)):
    if df['Gap'].iloc[i] > 0:  
        df.loc[df.index[i], 'Gap_Filled'] = any(df['Low'].iloc[i:] <= df['Close'].iloc[i-1])
    elif df['Gap'].iloc[i] < 0:  
        df.loc[df.index[i], 'Gap_Filled'] = any(df['High'].iloc[i:] >= df['Close'].iloc[i-1])

count_true = (df['Gap_Filled']=='True').sum()
count_false = (df['Gap_Filled']=='False').sum()
count_all = (df['Gap_Filled']).count()

print([gap_size_avg,gap_size_std,count_true,count_false,count_all])
df
print(df['Gap_Filled'].unique())

```

    [0.26470141771780065, 0.3128823308305326, 0, 0, 252]
    [False  True]

