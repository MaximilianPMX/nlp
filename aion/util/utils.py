def load_sentences(file_path):
    # Load sentences from a file, splitting by lines
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    return sentences
