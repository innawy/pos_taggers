import numpy as np
import time
from pandas import read_csv
import pandas as pd
from probs import transition, emission, build_emission_map, build_transition_map
from data import build_sentences_labels, build_sentences, handle_uncommon_words, handle_unknown_sentences, get_vocab_sentences, handle_unknown_words_test, cached
FILEPATH = '/Users/wanyinlin/Google Drive/a_NLP/assignment-3-nlpanthera-master/data'


@cached('generate_suffix_dict_cached.pickle')
def generate_suffix_dict(X_train, y_train, n_list=[3, 4, 5], prob_thres=.8, count_thres=100):
    '''
    last n letters in a word
    possible adds: top 2 prob instead of 1, may be marginal
                   min_appear: min num of appearance of a certain suffix
    '''
    word_train = list(X_train['word'])
    label_train = list(y_train['tag'])
    suff_dict_alln = {}
    for n in n_list:
        print('Generating Suffix Dict', n)
        suff = {}

        all_labels = set()
        for label in label_train:
            all_labels.add(label)
        label_list = sorted(all_labels)

        for j, word in enumerate(word_train):
            if j == 0:
                continue

            if len(word) < n+1:
                continue
            else:
                # print (word[-n:], labels[i][j])
                if word[-n:] in suff.keys():
                    suff[word[-n:]][label_list.index(label_train[j])] += 1
                else:
                    suff[word[-n:]] = np.zeros(len(label_list))
                    suff[word[-n:]][label_list.index(label_train[j])] = 1

        suff_dict = {}
        for s in suff.keys():
            if suff[s][np.argmax(suff[s])]/sum(suff[s]) >= prob_thres and sum(suff[s]) > count_thres:
                sorted_suffs = np.argsort(-np.array(suff[s]))
                suff_dict[s] = (label_list[sorted_suffs[0]],
                                suff[s][sorted_suffs[0]]/sum(suff[s]))
        # print(len(suff_dict))
        suff_dict_alln.update(suff_dict)

    return suff_dict_alln


def check_suffix(word, suff_dict, n_list=[3, 4, 5]):
    '''
    getting the top label with prob 1
    '''
    max_prob = 0
    max_label = None
    for n in n_list:
        if len(word) < n:
            continue
        elif word[-n:] in suff_dict.keys():
            if max_prob < suff_dict[word[-n:]][1]:
                max_prob = max(max_prob, suff_dict[word[-n:]][1])
                max_label = suff_dict[word[-n:]][0]
    if max_label == None:
        return None
    return [(max_label, max_prob)]


"""

X_train = pd.read_csv(FILEPATH+"/train_x.csv")
y_train = pd.read_csv(FILEPATH+"/train_y.csv")
#sentences, labels = build_sentences_labels(X_train, Y_train, k=1)
#sentences = handle_uncommon_words(sentences)


#suff_dict = generate_suffix_dict(X_train, y_train, n_list = [3,4,5])

#print (len(suff_dict))
#print (suff_dict.keys())
"""
