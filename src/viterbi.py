import time
import numpy as np
from probs import emission, transition, build_emission_map, build_transition_map
from pandas import read_csv
from data import build_sentences_labels, build_sentences, handle_uncommon_words, get_vocab_sentences, handle_unknown_sentences
from suffix_beam_accu import compAccu, check_suffix, generate_suffix_dict


def viterbi(sentences_dev, emissionMap, transitionMap, suff_dict):
    start = time.time()
    final = ['O']

    for si, sentence in enumerate(sentences_dev):
        # array of tag -> tag by column
        trellis = [{
            '<START>': 1
        }]
        # prev_scores = {}

        if si % 500 == 0:
            _time = time.time()-start
            print("{}/{}, time: {:4f}".format(si, len(sentences_dev), _time))
            start = time.time()

        backpointers = []
        prev_transitions = [1.0]

        for idx in range(1, len(sentence)):
            word = sentence[idx]
            mc = {}
            backpointers.append(mc)

            prev_word = sentence[idx-1]
            tag_emis = emission(word, emissionMap)
            if tag_emis == []:
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
                tag_emis = check_suffix(sentence[idx], suff_dict)
                if tag_emis == None:
                    tag_emis = [('NN', 1)]
                #print('emis_list from suffix', tag_emis)
                """
                if tag_emis == None:
                    unk_class_word = handle_uncommon_words(
                        [[sentence[idx]]])[0][0]
                    tag_emis = emission(unk_class_word, emission_map)
                """

            prev_column = trellis[-1]
            curr_column = {}

            for tag, emi in tag_emis:
                mx = -float("inf")
                mx_val = None
                mx_prb = None

                for prev_tag in prev_column.keys():
                    prev_trans_prb = prev_transitions[idx-1]

                    transition_prb = transition(
                        prev_tag, tag, transitionMap, smooth_type='add-k', prev_tag_prb=prev_trans_prb, smooth_factor=0.40)

                    score = prev_column[prev_tag] + \
                        np.log(transition_prb * emi)
                    if mx_val is None or score > mx:
                        mx_val = prev_tag
                        mx = score
                        mx_prb = transition_prb

                curr_column[tag] = mx
                backpointers[-1][tag] = mx_val
                prev_transitions.append(mx_prb)

            trellis.append(curr_column)
        backpointers.reverse()
        prev = backpointers[0]['<END>']
        result = []
        for mp_idx in range(1, len(backpointers)):
            result.append(prev)
            mp = backpointers[mp_idx][prev]
            #print(f"{prev} -> {mp}")
            prev = mp
        result.reverse()
        final = final + result
    return final


X_train = read_csv("./data/train_x.csv")
Y_train = read_csv("./data/train_y.csv")
sentences, labels = build_sentences_labels(X_train, Y_train, k=1)
# sentences = handle_uncommon_words(sentences, threshold=3)
vocab = get_vocab_sentences(sentences)

emission_map = build_emission_map(sentences, labels)
transition_map = build_transition_map(labels)

X_test = read_csv("./data/dev_x.csv")
y_dev = read_csv("./data/dev_y.csv")
sentences_test = build_sentences(X_test, k=1)
# sentences_test = handle_unknown_sentences(sentences_test, vocab)

suffix_dict = generate_suffix_dict(X_train, Y_train)

pred_dev = viterbi(sentences_test, emission_map, transition_map, suffix_dict)
print('Accuracy:', compAccu(X_test, y_dev, pred_dev, vocab))

print(emission("'s", emission_map))
