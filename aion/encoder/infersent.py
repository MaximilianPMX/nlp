from aion.encoder.infersent_lib import models
import torch

class InferSent:
    def __init__(self, version=1, params=None):
        # Load the correct model version
        self.version = version
        if version == 1:
            self.model = models.InferSent(params)
        elif version == 2:
            self.model = models.InferSent(params)
        else:
            raise ValueError("Version must be 1 or 2")

    def load_state_dict(self, state_dict):
        # Load pre-trained model weights
        self.model.load_state_dict(state_dict)

    def set_w2v_path(self, path_to_w2v):
        # Set the path to the word vectors
        self.model.set_w2v_path(path_to_w2v)

    def build_vocab(self, sentences, tokenize=False):
        # Build vocabulary from sentences
        self.model.build_vocab(sentences, tokenize=tokenize)

    def encode(self, sentences, bsize=64, tokenize=False, verbose=False):
        # Encode sentences into embeddings
        return self.model.encode(sentences, bsize=bsize, tokenize=tokenize, verbose=verbose)

    def visualize(self, emb, sentences):
        # Visualize embeddings (not implemented in the original code)
        # This function would allow for t-SNE visualization of the embeddings
        pass
