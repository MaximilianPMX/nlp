class InferSentDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.sentences = self.load_sentences()

    def load_sentences(self):
        sentences = []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    sentences.append(line.strip())
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]