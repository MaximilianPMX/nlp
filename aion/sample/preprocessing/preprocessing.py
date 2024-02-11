import nltk
import re


class Preprocessor:
    def __init__(self, lower=True, tokenize=True):
        self.lower = lower
        self.tokenize = tokenize

    def preprocess(self, text):
        if self.lower:
            text = text.lower()
        if self.tokenize:
            text = self.tokenize_text(text)
        return text

    def tokenize_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        return tokens

if __name__ == '__main__':
    # Simple test case
    processor = Preprocessor()
    text = "This is a Sample Text!"
    preprocessed_text = processor.preprocess(text)
    print(f"Original text: {text}")
    print(f"Preprocessed text: {preprocessed_text}")
