# Imported Libraries
import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify time period for data
end_date = datetime.now().strftime('%Y-%m-%d')  # Today
start_date = '2021-01-28'
end_date = '2021-08-16'
interval = '1d'

# Getting stock market data for S&P 500
stock = yf.Ticker("amc")
stock_hist = stock.history(start=start_date, end=end_date, interval=interval)

# Reset index to move Date from index to column
stock_hist.reset_index(inplace=True)

# Ensure Date is in consistent format 'YYYY-MM-DD' (to match GME data)
stock_hist['Date'] = pd.to_datetime(stock_hist['Date']).dt.strftime('%Y-%m-%d')

# Print the reformatted data
print(stock_hist.head())

# Plotting closing price over specified time
plt.figure(figsize=(14, 8))
sns.lineplot(data=stock_hist, x='Date', y='Close', label='Closing Price', color='b')
plt.xlabel('Date')
plt.title('AMC Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save stock data as CSV
stock_hist.to_csv("amc stock.csv", index=False)
