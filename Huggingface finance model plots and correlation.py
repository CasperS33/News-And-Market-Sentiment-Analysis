# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load Sentiment Data ---
# Load financial sentiment data
sentiment_df = pd.read_csv('financial_sentiment_with_timestamps.csv')

# Define keywords for filtering
keywords = ['GME', 'GameStop', 'gamestop', 'Gamestop', 'gme']

# Map sentiment categories to numeric values
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
sentiment_df['numeric_sentiment'] = sentiment_df['sentiment'].map(sentiment_mapping)

# Drop rows where 'timestamp' or 'numeric_sentiment' is NaN
sentiment_df.dropna(subset=['timestamp', 'numeric_sentiment'], inplace=True)

# Convert 'timestamp' to datetime format
sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])

# Filter the data for the specified keywords in 'combined_text'
filtered_sentiment = sentiment_df[
    sentiment_df['combined_text'].str.contains('|'.join(keywords), case=False, na=False)
]

# Calculate the average sentiment score for each day
filtered_sentiment.set_index('timestamp', inplace=True)
average_daily_sentiment = filtered_sentiment.resample('D')['numeric_sentiment'].mean()

# Interpolate missing values in average daily sentiment
average_daily_sentiment = average_daily_sentiment.interpolate()

# --- Load and Process Stock and Market Data ---
# Load the GME stock data
gme = pd.read_csv('Exam files folder/Stock market data/amc stock.csv')
sp500 = pd.read_csv('sp500_stock_prices.csv')

# Ensure 'Date' columns are in datetime format
gme['Date'] = pd.to_datetime(gme['Date'], utc=True)
sp500['Date'] = pd.to_datetime(sp500['Date'], utc=True)

# Calculate Relative Change (Log Returns) for GME
gme['Rel_change'] = np.log(gme['Close']) - np.log(gme['Open'])

# Calculate Relative Change (Log Returns) for SP500
sp500['Rel_change'] = np.log(sp500['Close']) - np.log(sp500['Open'])

# Merge GME and SP500 data on 'Date'
combined_data = pd.merge(gme, sp500, on='Date', suffixes=('_GME', '_SP500'))

# Calculate GME relative change to SP500
combined_data['GME_rel_change_to_SP500'] = combined_data['Rel_change_GME'] - combined_data['Rel_change_SP500']

# --- Combine Sentiment Data with Stock Data ---
# Reset index for sentiment data and rename columns
average_daily_sentiment = average_daily_sentiment.reset_index()
average_daily_sentiment.rename(columns={'timestamp': 'Date', 'numeric_sentiment': 'Avg_sentiment_score'}, inplace=True)

average_daily_sentiment['Date'] = average_daily_sentiment['Date'].dt.tz_localize('UTC')

# Merge average sentiment with stock data
combined_data = pd.merge(combined_data, average_daily_sentiment, on='Date', how='left')

# Interpolate missing values in combined_data columns
combined_data['Avg_sentiment_score'] = combined_data['Avg_sentiment_score'].interpolate()
combined_data['GME_rel_change_to_SP500'] = combined_data['GME_rel_change_to_SP500'].interpolate()

# --- Correlation Analysis ---
# Add time lag for Avg. Sentiment Score
time_lag = 1  # Change this value to experiment with different lags (in days)
combined_data['Lagged_sentiment'] = combined_data['Avg_sentiment_score'].shift(time_lag)

# Ensure the data is clean and aligned for correlation analysis
correlation_data_lagged = combined_data[['Lagged_sentiment', 'GME_rel_change_to_SP500']].dropna()

# Compute Pearson correlation coefficient for the lagged data
correlation_coefficient_lagged = correlation_data_lagged.corr().iloc[0, 1]

# Print the results
print(f"Pearson correlation coefficient (Lagged by {time_lag} day): {correlation_coefficient_lagged:.2f}")

# Optionally, test for multiple time lags
for lag in range(-3, 8):  # Testing for lags from 1 to 7 days
    combined_data[f'Lagged_sentiment_{lag}'] = combined_data['Avg_sentiment_score'].shift(lag)
    corr_data = combined_data[[f'Lagged_sentiment_{lag}', 'GME_rel_change_to_SP500']].dropna()
    if not corr_data.empty:
        corr_value = corr_data.corr().iloc[0, 1]
        print(f"Lag {lag} days: Correlation coefficient = {corr_value:.2f}")

# --- Plotting Combined Graphs ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Avg. Sentiment Score
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Avg. Sentiment Score', color=color)
ax1.plot(combined_data['Date'], combined_data['Avg_sentiment_score'], label='Avg Sentiment Score (GME)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for GME relative change to SP500
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('GME Rel. Change to SP500', color=color)
ax2.plot(combined_data['Date'], combined_data['GME_rel_change_to_SP500'], label='AMC Rel. Change to SP500', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add titles and legend
fig.suptitle('Avg. Sentiment Score (GME) and GME Relative Change to SP500', fontsize=14)
fig.tight_layout()
plt.show()
