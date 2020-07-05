import os
import pickle
import pandas as pd
from word_classes import get_word_class, get_word_class_test
FILEPATH = '/Users/wanyinlin/Google Drive/a_NLP/assignment-3-nlpanthera-master/data'
START_TOKEN = "<START>"
END_TOKEN = "<END>"


def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """
    def decorator(fn):  # define a decorator for a function "fn"
        # define a wrapper that will finally call "fn" with all arguments
        def wrapped(*args, **kwargs):
            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as cachehandle:
                    print("using cached result from '%s'" % cachefile)
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("saving result to cache '%s'" % cachefile)
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator   # return this "customized" decorator that uses "cachefile"


def read_csv(path):
    df = pd.read_csv(path)
    df.set_index('id')
    return df


@cached('build_sentences_cache.pickle')
def build_sentences(words_df, k):
    sentences = []
    current_sentence = [START_TOKEN]*k
    for index, row in words_df.iterrows():
        if index % 5000 == 0:
            print("Build sentences: {} / {}".format(index, len(words_df)))
        if index != 0:
            word = row.word

            current_sentence.append(word)

            if (word == "." or word == "?" or word == "!"):
                current_sentence.append(END_TOKEN)
                sentences.append(current_sentence)
                current_sentence = [START_TOKEN]*k

    return sentences


@cached('build_sentences_labels_cache.pickle')
def build_sentences_labels(words_df, labels_df, k):
    sentences = []
    labels = []

    current_sentence = [START_TOKEN]*k
    current_label = [START_TOKEN]*k

    for index, row in words_df.iterrows():
        if index % 5000 == 0:
            print("Build sentence & labels: {} / {}".format(index, len(words_df)))
        if index != 0:
            word = row.word
            label = labels_df.loc[index].tag

            current_sentence.append(word)
            current_label.append(label)

            if (word == "." or word == "?" or word == "!"):
                current_label.append(END_TOKEN)
                current_sentence.append(END_TOKEN)
                sentences.append(current_sentence)
                labels.append(current_label)
                current_sentence = [START_TOKEN]*k
                current_label = [START_TOKEN]*k

    return sentences, labels


def get_vocab(df):
    vocab = set()
    for _, row in df.iterrows():
        vocab.add(row.word)

    return vocab


def get_vocab_sentences(sentences):
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    return vocab


def handle_uncommon_words(sentences, threshold=5):  # for train
    """Handles uncommon words by finding words that have been found < threshold times in sentences and then maps it to a
    smaller set of classes based on word_classes
    Inputs:
        sentences: string[] => A set of sentences to map uncommon words to classes for
        threshold: number => How many times does a word have to appear before it is considered "common"
    """

    word_counts = {}

    for sentence in sentences:
        for word in sentence:
            count = 0
            if (word in word_counts):
                count = word_counts[word]
            count += 1
            word_counts[word] = count

    handled = 0

    for sentence_idx, sentence in enumerate(sentences):
        for word_idx, word in enumerate(sentence):
            if (word not in word_counts or word_counts[word] < threshold):
                handled += 1
                clz = get_word_class(word)
                sentences[sentence_idx][word_idx] = clz
    #print('handled train: ', handled)
    return sentences


def is_unk(word, vocab):
    return (word in vocab)


def handle_unknown_sentences(sentences, vocab):  # for dev and test
    unk_count = 0
    for sentence_idx, sentence in enumerate(sentences):
        for word_idx, word in enumerate(sentence):
            if word not in vocab:
                unk_count += 1
                clz = get_word_class_test(word)
                sentences[sentence_idx][word_idx] = clz
    print('unk count in dev/test:', unk_count)
    return sentences


def handle_unknown_words(list_of_unknown):
    unk_label = {}
    for word in list_of_unknown:
        clz = get_word_class(word)
        unk_label[word] = clz

    return unk_label


def handle_unknown_words_test(list_of_unknown):
    unk_label = {}
    for word in list_of_unknown:
        clz = get_word_class_test(word)
        unk_label[word] = clz

    return unk_label
