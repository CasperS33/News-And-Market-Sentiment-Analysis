if __name__ == '__main__':
    # Install the required libraries if not already installed 
    # !pip install transformers nltk pandas matplotlib seaborn
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from transformers import pipeline
    import nltk
    import pandas as pd
    import re
    import string
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure VADER lexicon is downloaded
    nltk.download('vader_lexicon')

    # Load your dataset
    file_path = 'reddit_wsb.csv'  
    data = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=',')

    # Process the timestamp column
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H.%M', errors='coerce').dt.strftime('%d/%m/%Y')

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

    # Concatenate 'title' and 'body' into a single column, treating missing or empty 'body' as an empty string
    data['combined_text'] = data['title'] + " " + data['body'].fillna("")

    # Truncate combined text for safety
    def truncate_text(text, max_words=250):
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text

    data['combined_text'] = data['combined_text'].apply(lambda x: truncate_text(x, max_words=250))

    # Extract columns for analysis
    timestamp_list = data['timestamp'].tolist()
    combined_text_list = data['combined_text'].tolist()

    # Financial-Social Sentiment Analysis
    print("Running Financial-Social sentiment analysis on 'combined_text'...")
    financial_social_model = pipeline(
        "text-classification", 
        model="soleimanian/financial-roberta-large-sentiment"
    )

    # Sentiment Analysis for 'combined_text'
    financial_results_combined = financial_social_model(
        combined_text_list, 
        truncation=True,     
        max_length=512      
    )

    financial_output_combined = pd.DataFrame({
        "combined_text": combined_text_list,
        "sentiment": [result['label'] for result in financial_results_combined],
        "confidence": [result['score'] for result in financial_results_combined],
        "timestamp": timestamp_list
    })
    print("Financial-Social Results for Combined Text:\n", financial_output_combined)

    # Save the results to a CSV file
    financial_output_combined.to_csv("financial_sentiment_results_combined.csv", index=False)
    print("Results saved to 'financial_sentiment_results_combined.csv'.")

    # Create histogram for combined_text
    print("Creating sentiment histogram for 'combined_text'...")

    # Map sentiment labels to numerical scores
    sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
    financial_output_combined['sentiment_score'] = financial_output_combined['sentiment'].map(sentiment_mapping)

    # Set Seaborn style
    sns.set(style="darkgrid")

    # Plot histogram for 'combined_text'
    plt.figure(figsize=(12, 6))
    sns.histplot(financial_output_combined['sentiment_score'], bins=3, kde=False, discrete=True)
    plt.title("Sentiment Distribution for Combined Text")
    plt.xlabel("Sentiment Score (Negative=-1, Neutral=0, Positive=1)")
    plt.ylabel("Frequency")
    plt.xticks(ticks=[-1, 0, 1], labels=["Negative", "Neutral", "Positive"])
    plt.savefig("financial_sentiment_histogram_combined.png")
    plt.show()

    print("Histogram saved as 'financial_sentiment_histogram_combined.png'.")
