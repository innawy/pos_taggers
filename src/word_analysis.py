from word_classes import get_word_class, word_classes
from collections import defaultdict
from pandas import read_csv
from data import build_sentences_labels, build_sentences
FILEPATH = '/Users/wanyinlin/Google Drive/a_NLP/assignment-3-nlpanthera-master/data'


def get_tag_counts(sentences, labels, word_class):
    tag_counts = defaultdict(int)

    for sentence, label_set in zip(sentences, labels):
        for word, label in zip(sentence, label_set):
            clz = get_word_class(word)
            if clz == word_class:
                tag_counts[label] += 1

    return tag_counts


def get_class_suffix_counts(sentences, labels, word_class):
    # for dev and test
    tag_counts = defaultdict(int)

    for sentence, label_set in zip(sentences, labels):
        for word, label in zip(sentence, label_set):
            clz = get_word_class(word)
            if clz == word_class:
                tag_counts[label] += 1

    return tag_counts

    tag_counts = defaultdict(int)

    for sentence, label_set in zip(sentences, labels):
        for word, label in zip(sentence, label_set):
            clz = get_word_class(word)
            if clz == word_class:
                tag_counts[label] += 1

    return tag_counts


X_train = read_csv(FILEPATH+"/train_x.csv")
Y_train = read_csv(FILEPATH+"/train_y.csv")
sentences, labels = build_sentences_labels(X_train, Y_train, k=1)


X_dev = read_csv(FILEPATH+"/dev_x.csv")
Y_dev = read_csv(FILEPATH+"/dev_y.csv")
sentences_dev, labels_dev = build_sentences_labels(X_dev, Y_dev, k=1)

X_test = read_csv(FILEPATH+"/test_x.csv")
sentences_test = build_sentences(X_test, k=1)


for clz in word_classes.keys():
    counts_train = get_tag_counts(sentences, labels, clz)
    print(clz)
    counts_dev = get_tag_counts(sentences_dev, labels_dev, clz)
    print(clz)
    print('-----------------')
    print(counts_dev)
    print('-----------------\n\n')
