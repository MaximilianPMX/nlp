import re

def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    # Simple tokenization (split by space)
    return text.split()


if __name__ == '__main__':
    # Example Usage
    sample_text = "This is a sample text with some special characters! and extra   spaces."
    cleaned_text = clean_text(sample_text)
    print("Cleaned text:", cleaned_text)
    tokens = tokenize(cleaned_text)
    print("Tokens:", tokens)
