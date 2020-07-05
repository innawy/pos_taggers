from pandas import read_csv
from data import build_sentences


def get_shallow_stats(x_path):
    X = read_csv(x_path)
    sentences = build_sentences(X, k=1)

    word_count = 0
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            word_count += 1
            vocab.add(word)

    return len(sentences), word_count, len(vocab)


print(get_shallow_stats("./data/test_x.csv"))
