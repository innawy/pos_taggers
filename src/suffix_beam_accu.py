import numpy as np
import time
from pandas import read_csv
import pandas as pd
from sklearn.metrics import confusion_matrix as cm
from suffix import generate_suffix_dict, check_suffix
from probs import transition, emission, build_emission_map, build_transition_map, build_transition_map_trigram
from data import is_unk, build_sentences_labels, build_sentences, handle_uncommon_words, handle_unknown_sentences, get_vocab_sentences, handle_unknown_words_test, cached
FILEPATH = '/Users/wanyinlin/Google Drive/a_NLP/assignment-3-nlpanthera-master/data'

# @cached('generate_suffix_dict_old.pickle')


def generate_suffix_dict_old(sentences, labels, n=5, topLabel=2):
    '''
    last n letters in a word
    possible adds: top 2 prob instead of 1, may be marginal
                   min_appear: min num of appearance of a certain suffix
    '''
    suff = {}

    all_labels = set()
    for label_set in labels:
        for label in label_set:
            all_labels.add(label)
    label_list = sorted(all_labels)

    for i, sentence in enumerate(sentences):
        if i % 5000 == 0:
            print('Generating Suffix Dict:', i)
        for j, word in enumerate(sentence):
            # print (i, j)
            if labels[i][j] == '<START>' or labels[i][j] == '<END>':
                continue
            if len(word) < n:
                continue
            else:
                # print (word[-n:], labels[i][j])
                if word[-n:] in suff.keys():
                    suff[word[-n:]][label_list.index(labels[i][j])] += 1

                else:
                    suff[word[-n:]] = np.zeros(len(label_list))
                    suff[word[-n:]][label_list.index(labels[i][j])] = 1

    suff_dict = {}
    for s in suff.keys():
        suff_dict[s] = label_list[np.argmax(suff[s])]

    return suff_dict


@cached('beam_search_dev_cached.pickle')
def beam_search(sentences, emission_map, transition_map, k=2, n=3):
    all_topk_seq = []
    emis_list_dummy = []
    start = time.time()
    for si, sentence in enumerate(sentences):
        if si % 1000 == 0:
            _time = time.time()-start
            print("Beam Searching: {}/{}, time: {:4f}".format(si, len(sentences), _time))
            start = time.time()
        topk_seq = [(['<START>']*(n-1), 1)]  # list of (seq_list, score)
        for idx in range(n-1, len(sentence)-1):
            # for tag_i in range(len(tags[]))
            # print (sentence[idx])
            seq_score = {}
            emis_list = emission(sentence[idx], emission_map)
            #trans_factor, emis_factor = 1, 1
            # emission output: [(tag1, prob1), (tag2, prob2)...]
            # print (emis_list)
            if emis_list == []:
                # for ki in range(len(topk_seq)):
                    # unk_tag =
                    # topk_seq[ki][0].append('NN')
                '''
                call the unknown word function to match it with a tag 
                by this time, before running the beam search, dev and test should have been cleaned up for digits, letters
                things left are capital, lowercase, other
                so check suffix first (check if suffix in suffix dict, if yes, match to label, generate new emis_list)
                if suffix does not work, label it as capital, lowercase, etc. run emission(word, emission_map) to get emis_list
                since train have cpaitla etc. so there should be two handle uncommon function for train and dev/test
                '''
                emis_list = check_suffix(sentence[idx], suff_dict)
                #print ('emis_list from suffix', emis_list)
                if emis_list == None:
                    unk_class_word = handle_uncommon_words(
                        [[sentence[idx]]])[0][0]
                    emis_list = emission(unk_class_word, emission_map)
                    #print('emis_list from classes', emis_list)
            # create possible sequence at each step given the previous top k sequence
            seq_possible = []
            for seq_prob in topk_seq:
                for i, tag_prob in enumerate(emis_list):
                    new_seq = []
                    new_seq.extend(seq_prob[0])
                    new_seq.append(tag_prob[0])
                    seq_possible.append((new_seq, seq_prob[1]))
            #print ('seq_possible', seq_possible)
            # Expected seq_possible = [(['<START>', 'N'],1), (['<START>', 'V'],1), (['<START>', 'J'],1)]

            topk_seq = []
            for seq_prob in seq_possible:
                emis = [t_p[1]
                        for t_p in emis_list if t_p[0] == seq_prob[0][-1]][0]
                # transition input:(tag_prev, tag2_current) output:prob
                if n == 2:
                    tran = transition(
                        seq_prob[0][-2], seq_prob[0][-1], transition_map)
                if n == 3:
                    tran = transition(
                        (seq_prob[0][-3], seq_prob[0][-2]), seq_prob[0][-1], transition_map)
                if n == 4:
                    tran = transition(
                        (seq_prob[0][-4], seq_prob[0][-3], seq_prob[0][-2]), seq_prob[0][-1], transition_map)
                prev_score = seq_prob[1]

                score = np.log(emis*tran)+prev_score
                # print (score)
                topk_seq.append((seq_prob[0], score))

            topk_seq.sort(key=lambda tup: tup[1])
            topk_seq = topk_seq[-k:]
            #print ('topk_seq', topk_seq)
        #print (top_topk_seq)
        top_topk_seq = topk_seq[-1][0]
        all_topk_seq.append(top_topk_seq)

    pred_dev_beam = ['O']
    for seq in all_topk_seq:
        for t in seq:
            if t == '<START>' or t == '<END>':
                continue
            else:
                pred_dev_beam.append(t)

    return pred_dev_beam


def correct_sentence(pred_labels, label_set):
    for pred, gold in zip(pred_labels, label_set):
        if pred != gold:
            return False
    return True


def compAccu(X_dev, y_dev_df, pred_y, vocab, k=1):
    X_def_list = list(X_dev['word'])
    y_dev = list(y_dev_df['tag'])
    correct = 0
    total = 0
    unk_count = 0
    incorrect = []
    unk_incorrect = []
    unk_incorrect_label = []
    known_incorrect = []
    # print (pred_y)
    sentence_labels = []
    sentence_preds = []
    optimal = 0
    suboptimal = 0
    for word, pred, truth in zip(X_def_list, pred_y, y_dev):
        # print (pred, dev)
        if truth == '*' or truth == '<STOP>':
            continue
        if not is_unk(word, vocab):
            unk_count += 1
        if pred == truth:
            correct += 1
        else:
            incorrect.append([word, pred, truth])
            if is_unk(word, vocab):
                known_incorrect.append([word, pred, truth])
            else:
                unk_incorrect.append([word, pred, truth])
                unk_incorrect_label.append(pred)
                unk_incorrect_label.append(truth)
        total += 1
        sentence_labels.append(truth)
        sentence_preds.append(pred)
        if word == '.' or word == '!' or word == '?':
            if not correct_sentence(sentence_preds, sentence_labels):
                suboptimal += 1
            else:
                optimal += 1
            sentence_labels = []
            sentence_preds = []
        accu = correct/total

    print(
        f"Suboptimal: {suboptimal} / {optimal + suboptimal} = {suboptimal / (optimal + suboptimal)}")

    incorrect_df = pd.DataFrame(incorrect, columns=['X', 'pred', 'truth'])
    unk_incorrect_df = pd.DataFrame(
        unk_incorrect, columns=['X', 'pred', 'truth'])
    unk_incorrect_df.to_csv('unk_incorrect.csv')
    #print (incorrect_df)
    #print (unk_incorrect_df)
    #print ('known_accu: ', 1-len(known_incorrect)/(total-unk_count))
    print('unk_accu: ', 1-len(unk_incorrect)/unk_count)
    conf = cm(unk_incorrect_df['truth'], unk_incorrect_df['pred'], labels=[
              'NN', 'NNS', 'NNP', 'NNPS'])
    np.savetxt("conf", conf, delimiter=",", fmt='%3.0f')
    return accu


"""
NGRAM =4

X_train = pd.read_csv(FILEPATH+"/train_x.csv")
Y_train = pd.read_csv(FILEPATH+"/train_y.csv")
sentences, labels = build_sentences_labels(X_train, Y_train, k=NGRAM-1)
sentences = handle_uncommon_words(sentences, 2)
vocab = get_vocab_sentences(sentences)
suff_dict = generate_suffix_dict(X_train, Y_train, n_list = [3,4,5], prob_thres=0.8, count_thres=100)

emission_map = build_emission_map(sentences, labels)
transition_map = build_transition_map(labels, n=NGRAM)


MODE = 'dev'

if MODE == 'dev':
    X_dev= read_csv(FILEPATH+"/dev_x.csv")
    y_dev = read_csv(FILEPATH+"/dev_y.csv")
    #print ([[list(X_dev['word'])[0]]])
    sentences_dev = build_sentences(X_dev, k=NGRAM-1)
    sentences_dev = handle_unknown_sentences(sentences_dev, vocab)

    pred_dev_beam = beam_search(sentences_dev, emission_map, transition_map, k=3, n=NGRAM)

    print('Accuracy:', compAccu(X_dev, y_dev, pred_dev_beam))

if MODE == 'test':
    X_test= read_csv(FILEPATH+"/test_x.csv")

    sentences_test = build_sentences(X_test, k=1)
    sentences_test = handle_unknown_sentences(sentences_test, vocab)

    pred_test_beam = beam_search(sentences_test, emission_map, transition_map, k=3, n=NGRAM)

    with open("test_y_0.csv", "w+") as f:
        f.write("id,tag\n")
        counter = 0
        for t in pred_test_beam:
            f.write(f"{counter},\"{t}\"\n")
            counter += 1



# emission_map = get_emission_map()
# beam_search()
# accu_beam = compAccu(X_dev, y_dev, pred_dev_beam)
# print(accu_beam)
"""
