import os
from aion.helper import text_helper
from aion.util import utils

class TextPreprocessor:
    def __init__(self):
        # Initialize TextPreprocessor
        pass

    def preprocess(self, text):
        # Preprocess the input text
        cleaned_text = text_helper.clean_text(text)
        return cleaned_text

    def tokenize(self, text):
        # Tokenize preprocessed text
        tokens = text_helper.tokenize(text)
        return tokens

if __name__ == '__main__':
    # Example Usage
    RESOURCES_PATH = 'aion/sample/resources'
    SAMPLE_TEXT_1 = os.path.join(RESOURCES_PATH, 'sample_text_1.txt')

    # Load sample sentences
    sentences = utils.load_sentences(SAMPLE_TEXT_1)
    sample_text = sentences[0]

    # Initialize TextPreprocessor
    preprocessor = TextPreprocessor()

    # Preprocess the text
    cleaned_text = preprocessor.preprocess(sample_text)
    print("Cleaned text:", cleaned_text)

    # Tokenize the text
    tokens = preprocessor.tokenize(cleaned_text)
    print("Tokens:", tokens)
