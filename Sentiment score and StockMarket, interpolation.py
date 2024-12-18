# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load Sentiment Data ---
# Load data from CSV
sentiment_df = pd.read_csv('Exam files folder/Sentiment results/vader_sentiment_results_combined.csv')

# Convert the 'timestamp' column to datetime
sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])

# Ensure the 'compound' sentiment score is numeric
sentiment_df['compound'] = pd.to_numeric(sentiment_df['compound'], errors='coerce')

# Drop rows where 'timestamp' or 'compound' is NaN
sentiment_df.dropna(subset=['timestamp', 'compound'], inplace=True)

# Set the 'timestamp' as the index
sentiment_df.set_index('timestamp', inplace=True)

# Filter data based on keywords in 'title'
keywords = ['GME', 'GameStop', 'gamestop', 'Gamestop', 'gme']
filtered_sentiment = sentiment_df[sentiment_df['combined_text'].str.contains('|'.join(keywords), case=False, na=False)]

# Calculate the average sentiment score for each day
average_daily_sentiment = filtered_sentiment.resample('D')['compound'].mean()

# Interpolate missing values in average_daily_sentiment
average_daily_sentiment = average_daily_sentiment.interpolate()

# --- Load and Process GME and SP500 Data ---
# Load the GME stock data
gme = pd.read_csv('Exam files folder/Stock market data/gme stock.csv')
sp500 = pd.read_csv('sp500_Stock_prices2.csv')

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
average_daily_sentiment.rename(columns={'timestamp': 'Date', 'compound': 'Avg_sentiment_score'}, inplace=True)

average_daily_sentiment['Date'] = average_daily_sentiment['Date'].dt.tz_localize('UTC')

# Merge average sentiment with stock data
combined_data = pd.merge(combined_data, average_daily_sentiment, on='Date', how='left')

# Interpolate missing values in combined_data columns
combined_data['Avg_sentiment_score'] = combined_data['Avg_sentiment_score'].interpolate()
combined_data['GME_rel_change_to_SP500'] = combined_data['GME_rel_change_to_SP500'].interpolate()

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
ax2.plot(combined_data['Date'], combined_data['GME_rel_change_to_SP500'], label='GME Rel. Change to SP500', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add titles and legend
fig.suptitle('Avg. Sentiment Score (GME) and GME Relative Change to SP500 (combined_text)', fontsize=14)
fig.tight_layout()
plt.show()
