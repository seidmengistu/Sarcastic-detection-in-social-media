import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from read_integrate_all_data_sources import read_all_data_sources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def analyze_source(df, title):

    df['text'] = df['text'].astype(str)

    print(f"\n📊 Analysis for: {title}")
    print("=" * (15 + len(title)))

    # Basic info
    print(f"Total Records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("Sample rows:")
    print(df.sample(3))

    # Nulls
    print("\nNull value count:")
    print(df.isnull().sum())

    # Class distribution
    print("\nClass distribution:")
    print(df['sarcastic'].value_counts(normalize=True).rename(
        lambda x: 'Sarcastic' if x == 1 else 'Not Sarcastic'))

    # Text length
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    print(f"\nAverage Text Length (chars): {df['char_count'].mean():.2f}")
    print(f"Average Text Length (words): {df['word_count'].mean():.2f}")

    # Top 10 frequent words
    words = []
    df['text'].str.lower().str.split().apply(
        lambda tokens: words.extend([w for w in tokens if w not in stop_words]))
    word_freq = Counter(words).most_common(10)
    print("\nTop 10 frequent words (excluding stopwords):")
    for word, freq in word_freq:
        print(f"{word}: {freq}")

    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='sarcastic', palette='viridis')
    plt.xticks([0, 1], ['Not Sarcastic', 'Sarcastic'])
    plt.title(f"{title} - Sarcasm Distribution")
    plt.show()

    # Plot histogram of word counts
    plt.figure(figsize=(6, 4))
    sns.histplot(df['word_count'], bins=40, color='skyblue')
    plt.title(f"{title} - Text Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.show()

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{title} - Word Cloud")
    plt.show()

    return df[["text", "sarcastic", "word_count", "char_count"]]


if __name__ == "__main__":
    reddit_data, headlines_data, sarcasm_corpus_combined, combined_data = read_all_data_sources()
    # reddit_stats = analyze_source(reddit_data, "Reddit Data")
    # headlines_stats = analyze_source(headlines_data, "Headlines JSON Data")
    # gen_stats = analyze_source(sarcasm_gen_data, "GEN Data")
    # hyp_stats = analyze_source(sarcasm_hyp_data, "HYP Data")
    rq_stats = analyze_source(sarcasm_corpus_combined,
                              "sarcasm_corpus_combined Data")
    # combined_stats = analyze_source(combined_data, "Combined Dataset")
