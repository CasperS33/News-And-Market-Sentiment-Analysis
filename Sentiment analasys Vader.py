if __name__ == '__main__':
    # Required libraries
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    import pandas as pd
    import re
    import string

    # Ensure VADER lexicon is downloaded
    nltk.download('vader_lexicon')

    # Load your dataset
    file_path = 'reddit_wsb.csv' 
    data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

    # Filter data for posts from 2021 and later
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data = data[data['timestamp'].dt.year >= 2021]

    # Clean the data
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()  # Convert to lowercase
            text = re.sub('@[^\s]+', '', text)  # Remove handlers
            text = re.sub(r"http\S+", "", text)  # Remove URLs
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            text = ' '.join(re.findall(r'\w+', text))  # Keep only words
            text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    # Clean 'title' and 'body' columns
    data['title'] = data['title'].apply(clean_text)
    data['body'] = data['body'].apply(clean_text)

    # Combine 'title' and 'body' into 'combined_text'
    data['combined_text'] = data.apply(
        lambda row: row['title'] if pd.isnull(row['body']) or row['body'] == "" 
        else row['title'] + " " + row['body'], axis=1)

    # Drop rows where combined_text or timestamp is missing
    data.dropna(subset=['combined_text', 'timestamp'], inplace=True)

    # Truncate text to 250 words
    data['combined_text'] = data['combined_text'].apply(lambda x: ' '.join(x.split()[:250]))

    # Initialize VADER Sentiment Analyzer
    print("Running VADER sentiment analysis on combined text...")
    vader_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [vader_analyzer.polarity_scores(text) for text in data['combined_text']]

    # Create DataFrame with sentiment results
    vader_output_combined = pd.DataFrame({
        "combined_text": data['combined_text'].tolist(),
        "positive": [score['pos'] for score in sentiment_scores],
        "neutral": [score['neu'] for score in sentiment_scores],
        "negative": [score['neg'] for score in sentiment_scores],
        "compound": [score['compound'] for score in sentiment_scores],
        "timestamp": data['timestamp'].tolist()
    })

    # Save results to CSV
    vader_output_combined.to_csv("vader_sentiment_results_combined.csv", index=False)
    print("Sentiment analysis results saved to 'vader_sentiment_results_combined.csv'.")

    # Plot Sentiment Distribution
    print("Creating sentiment distribution plot...")
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))

    # Calculate mean sentiment scores
    sentiment_means = vader_output_combined[['positive', 'neutral', 'negative']].mean()

    # Plot bar chart
    sentiment_means.plot(kind='bar')
    plt.title("Mean Sentiment Distribution for Combined Text")
    plt.ylabel("Mean Sentiment Score")
    plt.xticks(ticks=[0, 1, 2], labels=['Positive', 'Neutral', 'Negative'], rotation=0)
    plt.tight_layout()
    plt.savefig("sentiment_distribution_combined.png")
    plt.show()

    print("Sentiment distribution plot saved as 'sentiment_distribution_combined.png'.")
