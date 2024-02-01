from aion.sample.preprocessing.preprocessing import preprocess_text
from aion.sample.embeddings.embeddings import generate_embeddings
from aion.encoder.infersent import InferSent

# Sample sentences
sentences = [
    "This is the first sentence.",
    "Here is another sentence to embed.",
    "A third sentence for demonstration."
]

# Preprocessing
print("Preprocessing sentences...")
preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]
print("Preprocessed sentences:", preprocessed_sentences)

# Load InferSent model (replace with your actual path)
print("Loading InferSent model...")
infersent = InferSent(model_version=1)
infersent.load_model(path='encoder/infersent1.pth') # replace
infersent.set_w2v_path(path='fasttext/crawl-300d-2M.vec') # replace
infersent.build_vocab(preprocessed_sentences, tokenize=False)

# Generate embeddings
print("Generating embeddings...")
embeddings = generate_embeddings(infersent, preprocessed_sentences)

# Print embeddings
print("Embeddings shape:", embeddings.shape)
print("First sentence embedding:", embeddings[0])
