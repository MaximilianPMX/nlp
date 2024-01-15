import os
import torch

from aion.encoder.infersent import InferSent
from aion.util import utils

class EmbeddingGenerator:
    def __init__(self, model_version=1, w2v_path=None, model_path=None):
        # Initialize embedding generator
        self.model_version = model_version
        self.w2v_path = w2v_path
        self.model_path = model_path
        self.infersent = None

    def load_model(self):
        # Load the InferSent model
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, # Changed from 4096 ( halved the lstm layer size to reduce memory requirements.)
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': self.model_version}
        self.infersent = InferSent(self.model_version, params_model)
        self.infersent.load_state_dict(torch.load(self.model_path))
        self.infersent.set_w2v_path(self.w2v_path)

    def build_vocab(self, sentences, tokenize=False):
        # Build vocabulary for the sentences
        self.infersent.build_vocab(sentences, tokenize=tokenize)

    def generate_embeddings(self, sentences):
        # Generate embeddings for the input sentences
        embeddings = self.infersent.encode(sentences, bsize=128, tokenize=False, verbose=False)
        return embeddings

if __name__ == '__main__':
    # Example Usage
    RESOURCES_PATH = 'aion/sample/resources'
    W2V_PATH = os.path.join(RESOURCES_PATH, 'glove.840B.300d.txt')
    MODEL_PATH = os.path.join(RESOURCES_PATH, 'infersent1.pth')
    SAMPLE_TEXT_1 = os.path.join(RESOURCES_PATH, 'sample_text_1.txt')
    SAMPLE_TEXT_2 = os.path.join(RESOURCES_PATH, 'sample_text_2.txt')

    # Load sample sentences from files
    sentences1 = utils.load_sentences(SAMPLE_TEXT_1)
    sentences2 = utils.load_sentences(SAMPLE_TEXT_2)
    sentences = sentences1 + sentences2

    # Initialize EmbeddingGenerator
    embedding_generator = EmbeddingGenerator(w2v_path=W2V_PATH, model_path=MODEL_PATH)

    # Load the model
    embedding_generator.load_model()

    # Build vocabulary
    embedding_generator.build_vocab(sentences)

    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(sentences)

    print("Embeddings shape:", embeddings.shape)
