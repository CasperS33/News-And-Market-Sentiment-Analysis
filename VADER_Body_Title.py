if __name__ == '__main__':
    # Install the required libraries if not already installed 
    # !pip install transformers nltk matplotlib seaborn
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

    # Clean the data
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()  # Convert to lowercase
            text = re.sub('@[^\s]+', '', text)  # Remove handlers
            text = re.sub(r"http\S+", "", text)  # Remove URLs
            text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            text = ' '.join(re.findall(r'\w+', text))  # Remove special characters
            text = re.sub(r'\s+[a-zA-Z]\s+', '', text)  # Remove single characters
            text = re.sub(r'\s+', ' ', text, flags=re.I)  # Substitute multiple spaces with a single space
        return text

    # Apply cleaning to the 'title' and 'body' columns
    data['title'] = data['title'].apply(clean_text)
    data['body'] = data['body'].apply(clean_text)
    data.dropna(subset=['body', 'title'], inplace=True)  # Drop rows where 'body' or 'title' is missing

    # max tokens
    def truncate_text(text, max_words=250):
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text
    data['body'] = data['body'].apply(lambda x: truncate_text(x, max_words=250))
    data['title'] = data['title'].apply(lambda x: truncate_text(x, max_words=50))  # Shorter limit for titles

    # Select a subset of the data for analysis
    subset_titles = data['title'].dropna().head(50).tolist()
    subset_bodies = data['body'].dropna().head(50).tolist()

    # VADER Sentiment Analysis for 'body'
    print("Running VADER sentiment analysis on 'body'...")
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_results_body = [vader_analyzer.polarity_scores(body) for body in subset_bodies]
    vader_output_body = pd.DataFrame({
        "body": subset_bodies,
        "positive": [result['pos'] for result in vader_results_body],
        "neutral": [result['neu'] for result in vader_results_body],
        "negative": [result['neg'] for result in vader_results_body],
        "compound": [result['compound'] for result in vader_results_body]
    })
    print("VADER Results for Body:\n", vader_output_body)

    # VADER Sentiment Analysis for 'title'
    print("Running VADER sentiment analysis on 'title'...")
    vader_results_title = [vader_analyzer.polarity_scores(title) for title in subset_titles]
    vader_output_title = pd.DataFrame({
        "title": subset_titles,
        "positive": [result['pos'] for result in vader_results_title],
        "neutral": [result['neu'] for result in vader_results_title],
        "negative": [result['neg'] for result in vader_results_title],
        "compound": [result['compound'] for result in vader_results_title]
    })
    print("VADER Results for Title:\n", vader_output_title)

    # Save the results to CSV files
    vader_output_body.to_csv("vader_sentiment_results_body.csv", index=False)
    vader_output_title.to_csv("vader_sentiment_results_title.csv", index=False)

    # Plot grouped bar charts for mean sentiment scores
    print("Creating grouped bar charts for sentiment distributions...")

    # Set Seaborn style for better visuals
    sns.set(style="darkgrid")

    # Calculate mean sentiment scores for 'body' and 'title'
    body_means = vader_output_body[['positive', 'neutral', 'negative']].mean()
    title_means = vader_output_title[['positive', 'neutral', 'negative']].mean()

    # Create a DataFrame for plotting
    sentiment_means = pd.DataFrame({
        'Title Sentiment': title_means,
        'Body Sentiment': body_means
    }).T  # Transpose to match the format for grouped bars

    # Plot grouped bar charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for Title
    sentiment_means.loc['Title Sentiment'].plot(kind='bar', ax=axes[0])
    axes[0].set_title("Title Sentiment Distribution")
    axes[0].set_ylabel("Mean Sentiment Score")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'], rotation=0)

    # Plot for Body
    sentiment_means.loc['Body Sentiment'].plot(kind='bar', ax=axes[1])
    axes[1].set_title("Body Sentiment Distribution")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(['Positive Sentiment', 'Neutral Sentiment', 'Negative Sentiment'], rotation=0)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("sentiment_grouped_bar_plots.png")
    plt.show()

    print("Results and grouped bar plots saved as 'sentiment_grouped_bar_plots.png'.")
