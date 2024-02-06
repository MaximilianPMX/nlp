from aion.encoder.infersent import InferSent
import torch

class EmbeddingGenerator:
    def __init__(self, model_path, w2v_path, device='cpu'):
        self.model_path = model_path
        self.w2v_path = w2v_path
        self.device = device
        self.model = None

    def load_model(self):
        try:
            V = 2
            MODEL_PATH = self.model_path
            W2V_PATH = self.w2v_path

            infersent = InferSent(V, model_path=MODEL_PATH)
            infersent.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(self.device)))
            infersent.set_w2v_path(W2V_PATH)
            infersent.build_vocab_k_words(K=100000)

            self.model = infersent.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def generate_embeddings(self, sentences):
        if self.model is None:
            print("Model not loaded. Please load the model first.")
            return None
        try:
            embeddings = self.model.encode(sentences, bsize=128, tokenize=False, verbose=False)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None


if __name__ == '__main__':
    # Example Usage (replace with your actual paths)
    model_path = 'encoder/infersent.allnli.pickle'
    w2v_path = 'encoder/fasttext.42B.300d.txt'
    
    embedding_generator = EmbeddingGenerator(model_path, w2v_path, device='cpu')
    embedding_generator.load_model()

    sentences = ['This is the first sentence.', 'This is the second sentence.']
    embeddings = embedding_generator.generate_embeddings(sentences)

    if embeddings is not None:
        print("Embeddings generated successfully:")
        print(embeddings)
