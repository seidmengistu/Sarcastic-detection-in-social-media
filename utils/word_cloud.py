import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os
from config import Config

def generate_word_cloud(save_path='word_cloud.png', by_class=True):
    """
    Generate word cloud from preprocessed dataset.
    Args:
        save_path: Path to save the word cloud image
        by_class: If True, generate separate word clouds for sarcastic and non-sarcastic texts
    """
    try:
        df = pd.read_csv(Config.PREPROCESSED_DATA_PATH)
        
        if by_class:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            sarc_text = ' '.join(df[df['class'] == 'sarc']['text'].astype(str))
            wordcloud_sarc = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100
            ).generate(sarc_text)
            
            ax1.imshow(wordcloud_sarc, interpolation='bilinear')
            ax1.axis('off')
            ax1.set_title('Sarcastic Texts', fontsize=16)
            
            nonsar_text = ' '.join(df[df['class'] == 'notsarc']['text'].astype(str))
            wordcloud_nonsar = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100
            ).generate(nonsar_text)
            
            ax2.imshow(wordcloud_nonsar, interpolation='bilinear')
            ax2.axis('off')
            ax2.set_title('Non-Sarcastic Texts', fontsize=16)
            
            plt.tight_layout(pad=3)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            all_text = ' '.join(df['text'].astype(str))
            wordcloud = WordCloud(
                width=1600, height=800,
                background_color='white',
                max_words=200
            ).generate(all_text)
            
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Word cloud saved to {save_path}")
        
        words = ' '.join(df['text'].astype(str)).split()
        word_freq = Counter(words).most_common(20)
        
        print("\nTop 20 most frequent words:")
        print("-" * 40)
        for word, freq in word_freq:
            print(f"{word:20} {freq:>10}")
            
    except Exception as e:
        print(f"Error generating word cloud: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_word_cloud('word_cloud_by_class.png', by_class=True)
    generate_word_cloud('word_cloud_all.png', by_class=False) 