import spacy
import re
import string
import emoji

# Load spaCy's English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Main preprocessing function


def preprocess_text(
    text,
    lowercase=True,
    remove_punctuation=False,
    remove_stopwords=False,
    remove_digits=True,
    lemmatize=True,
    strip_whitespace=True,
    tokenize=False,
    replace_elongated=True,
    preserve_emojis=True,
):
    if replace_elongated:
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if preserve_emojis:
        text = emoji.demojize(text)
    # 1. Lowercase
    if lowercase:
        text = text.lower()

    # 2. Remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # 3. Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Strip extra whitespace
    if strip_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    # 5. Tokenize with spaCy
    doc = nlp(text)

    # 6. Lemmatize & Remove stopwords
    processed = []
    for token in doc:
        if remove_stopwords and token.is_stop:
            continue
        word = token.lemma_ if lemmatize else token.text
        processed.append(word)

    # 7. Return as string or list of tokens
    return processed if tokenize else ' '.join(processed)


# Example usage
if __name__ == "__main__":
    sample_text = "This is an example sentence, showing off the stopwords & punctuation removal!! 123"
    cleaned = preprocess_text(
        sample_text,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        remove_digits=True,
        lemmatize=True,
        strip_whitespace=True,
        tokenize=False  # Set to True to get a list of tokens
    )
    print("Processed text:", cleaned)
