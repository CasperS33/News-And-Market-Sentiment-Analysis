import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
import nltk

# Download punkt if not already installed
nltk.download('punkt_tab')

# Definess the word cloud function
def show_wordcloud(data, title="", max_words=100):
    text = " ".join(t for t in data.dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "fuck", "fucking"])
    
    # Set width, height, and limit the number of words
    wordcloud = WordCloud(
        stopwords=stopwords,
        width=400,  # Set width
        height=200,  # Set height
        max_font_size=50,
        max_words=max_words,  # Adjusted maximum number of words
        background_color="black"
    ).generate(text)
    
    # Ensure the figure size matches the word cloud aspect ratio
    fig = plt.figure(figsize=(10, 5)) 
    plt.axis('off')
    fig.suptitle(title, fontsize=20)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

# Reads the data file from reddit
Reddit_data = pd.read_csv('reddit_wsb.csv')
print(Reddit_data.columns)
print(Reddit_data.info)


# Converts timstamp to datetime
Reddit_data['timestamp'] = pd.to_datetime(Reddit_data['timestamp'])

# Creates week-day column for graph
day_of_the_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
days_order = list(day_of_the_week.values())
Reddit_data['Weekday'] = Reddit_data['timestamp'].apply(lambda x: day_of_the_week[x.weekday()])

# visualizes week day post
xs = Reddit_data['Weekday'].value_counts().index
ys = Reddit_data['Weekday'].value_counts().values

plt.figure(figsize=(14, 6))
sns.barplot(x=xs, y=ys, order=days_order)
plt.title("No. of Posts vs Day of the Week", fontsize=15)
plt.xlabel("Days", fontsize=15)
plt.ylabel("No. of Posts", fontsize=15)
plt.show()

# Cleans the data
reddit_title = Reddit_data['title'].dropna()
reddit_body = Reddit_data['body'].dropna()

def clean_text_date(text):
    text = text.lower()
    text = re.sub('@[^\s]+', '', text)  # Fjern mentions
    text = re.sub(r"http\S+", "", text)  # Fjern URLs
    text = ' '.join(re.findall(r'\w+', text))  # Fjern specialtegn
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Fjern enkeltbogstaver
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Fjern ekstra mellemrum
    return text

# Cleans the columns
reddit_title = reddit_title.apply(clean_text_date)
reddit_body = reddit_body.apply(clean_text_date)

# Calculates tht lenght of the title and body column
title_length = reddit_title.apply(lambda x: len(word_tokenize(x)))
body_length = reddit_body.apply(lambda x: len(word_tokenize(x)))

# Creates Histograms
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(title_length, bins=50, kde=True, ax=axis1, color="blue")
sns.histplot(body_length, bins=50, kde=True, ax=axis2, color="green")

axis1.set_title("Distribution of Title Lengths")
axis1.set_xlabel("Number of Words in Title")
axis1.set_ylabel("Frequency")

axis2.set_title("Distribution of Body Lengths")
axis2.set_xlabel("Number of Words in Body")
axis2.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Generate WordCloud for the title column
show_wordcloud(reddit_title, title="Word Cloud for Titles")

# Generate WordCloud for the body column
show_wordcloud(reddit_body, title="Word Cloud for Body Texts")
