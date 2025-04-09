#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from datetime import datetime
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/sales5.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfYear'] = data['Date'].dt.dayofyear
data['TotalItems'] = data['NumberGrocery'] + data['NumberNongrocery']
data['AvgPrice'] = (data['NumberGrocery']*data['PriceGrocery'] + data['NumberNongrocery']*data['PriceNongrocery'])/data['TotalItems']

plt.figure(figsize=(14, 7))
monthly_avg = data.groupby('Month')['TotalItems'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
bars = plt.bar(months, monthly_avg, color='#1f77b4', alpha=0.7, width=0.6, edgecolor='black', linewidth=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(round(height))}', ha='center', va='bottom', fontsize=9)

data_2022 = data[data['Year'] == 2022].copy()
daily_items_2022 = data_2022.groupby('DayOfYear')['TotalItems'].sum()
if len(daily_items_2022) < 365:
    daily_items_2022 = np.pad(daily_items_2022, (0, 365-len(daily_items_2022)), 'constant')
norm_factor = max(daily_items_2022)/max(monthly_avg)
daily_items_normalized = daily_items_2022/norm_factor

fft_coeffs = fft(daily_items_normalized)
fft_coeffs[8:-8] = 0
reconstructed = np.real(ifft(fft_coeffs))

x_vals = np.linspace(0, 11, 365)
plt.plot(x_vals, reconstructed, 'r-', linewidth=2)

plt.title('Superstore Monthly Sales Analysis\nStudent ID: 24082155', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Daily Items Sold', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('Figure1_Monthly_Sales.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(14, 7))
daily_data = data.groupby('Date').agg({'TotalItems':'sum', 'AvgPrice':'mean'}).reset_index()
plt.scatter(daily_data['TotalItems'], daily_data['AvgPrice'], alpha=0.6, color='green', s=50)

X = daily_data['TotalItems'].values.reshape(-1, 1)
y = daily_data['AvgPrice'].values.reshape(-1, 1)
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
plt.plot(X, y_pred, 'r-', linewidth=2)

avg_price_by_day = data.groupby('DayOfYear')['AvgPrice'].mean()
X_day = avg_price_by_day.idxmax()
Y_day = avg_price_by_day.idxmin()

plt.text(0.72, 0.93, f"X (Highest Price Day): {X_day}\nY (Lowest Price Day): {Y_day}",
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=11)

plt.title('Daily Price vs Sales Volume Analysis\nStudent ID: 24082155', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Number of Items Sold', fontsize=12)
plt.ylabel('Average Price (€)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('Figure2_Price_vs_Items.png', dpi=300, bbox_inches='tight')

print(f"X (Highest Price Day): {X_day}")
print(f"Y (Lowest Price Day): {Y_day}")
print(f"Day {X_day} → {datetime(2022, 1, 1) + pd.Timedelta(days=X_day-1):%b %d}")
print(f"Day {Y_day} → {datetime(2022, 1, 1) + pd.Timedelta(days=Y_day-1):%b %d}")

plt.show()

