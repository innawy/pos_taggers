from data import read_csv, build_sentences_labels, handle_uncommon_words, handle_unknown_words, build_sentences, handle_unknown_sentences
from probs import build_emission_map, build_transition_map


if __name__ == '__main__':
    X_train = read_csv("./data/dev_x.csv")
    X_labels = read_csv("./data/dev_y.csv")

    sentences, labels = build_sentences_labels(X_train, X_labels, k=2)
    sentences = handle_uncommon_words(sentences)

    #transition_map = build_transition_map(labels)
    #emission_map = build_emission_map(sentences, labels)
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)

    not_found = []
    test_sentences = build_sentences(read_csv('./data/test_x.csv'), k=2)
    test_sentences = handle_unknown_sentences(test_sentences, vocab)
    for test_sentence in test_sentences:
        for test_word in test_sentence:
            if test_word not in vocab:
                not_found.append(test_word)

    with open('output.txt', 'w') as f:
        for word in not_found:
            f.write(f"{word}\n")
