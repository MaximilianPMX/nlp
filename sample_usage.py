import os
from aion.sample.embeddings.embeddings import EmbeddingGenerator
from aion.sample.preprocessing.preprocessing import TextPreprocessor
from aion.util import utils

# Define paths to resources
RESOURCES_PATH = 'aion/sample/resources'
W2V_PATH = os.path.join(RESOURCES_PATH, 'glove.840B.300d.txt')
MODEL_PATH = os.path.join(RESOURCES_PATH, 'infersent1.pth')
SAMPLE_TEXT_1 = os.path.join(RESOURCES_PATH, 'sample_text_1.txt')
SAMPLE_TEXT_2 = os.path.join(RESOURCES_PATH, 'sample_text_2.txt')

# Load sample sentences
sentences1 = utils.load_sentences(SAMPLE_TEXT_1)
sentences2 = utils.load_sentences(SAMPLE_TEXT_2)
sentences = sentences1 + sentences2

# --- Text Preprocessing ---#

# Initialize the text preprocessor
text_preprocessor = TextPreprocessor()

# Preprocess the sentences
cleaned_sentences = [text_preprocessor.preprocess(sentence) for sentence in sentences]

# Tokenize the sentences
tokenized_sentences = [text_preprocessor.tokenize(sentence) for sentence in cleaned_sentences]

# Print some sample preprocessed and tokenized sentences
print("Original sentence:", sentences[0])
print("Cleaned sentence:", cleaned_sentences[0])
print("Tokenized sentence:", tokenized_sentences[0])

# --- Embedding Generation --- #

# Initialize the embedding generator
embedding_generator = EmbeddingGenerator(w2v_path=W2V_PATH, model_path=MODEL_PATH)

# Load the InferSent model
embedding_generator.load_model()

# Build the vocabulary
embedding_generator.build_vocab(sentences)

# Generate embeddings for the cleaned sentences
embeddings = embedding_generator.generate_embeddings(cleaned_sentences)

# Print the shape of the embeddings
print("Embeddings shape:", embeddings.shape)
