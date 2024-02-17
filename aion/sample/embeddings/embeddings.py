from aion.dataset.infersent_dataset import InferSentDataset
from aion.encoder.infersent import InferSent
import torch

class InferSentEmbeddings:
    def __init__(self, data_path, infersent_model_path, w2v_path):
        self.dataset = InferSentDataset(data_path)
        self.sentences = self.dataset.load_sentences()
        self.model = InferSent(infersent_model_path)
        self.model.load_state_dict(torch.load(infersent_model_path))
        self.model.set_w2v_path(w2v_path)
        self.model.build_vocab(self.sentences, tokenize=False)

    def generate_embeddings(self, sentences):
        return self.model.encode(sentences, tokenize=False)

if __name__ == '__main__':
    # Example Usage
    data_path = 'resources/sentences.txt'  # Replace with your data path
    infersent_model_path = 'resources/infersent1.pth'  # Replace with your InferSent model path
    w2v_path = 'resources/glove.840B.300d.txt' # Replace with your word2vec path

    embedding_generator = InferSentEmbeddings(data_path, infersent_model_path, w2v_path)
    sentences = ["This is the first sentence.", "This is the second sentence."]
    embeddings = embedding_generator.generate_embeddings(sentences)

    print(f"Embeddings shape: {embeddings.shape}")
