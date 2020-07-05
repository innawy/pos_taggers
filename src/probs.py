import pickle
from collections import defaultdict


def build_transition_map(labels, n=2):
    labels_ending = {}

    for sentence in labels:
        for label_idx, label in enumerate(sentence):
            if label_idx > n - 2:
                if n == 2:
                    prev_label = sentence[label_idx - 1]
                if n == 3:
                    prev_label = (sentence[label_idx-2], sentence[label_idx-1])
                if n == 4:
                    prev_label = (
                        sentence[label_idx-3], sentence[label_idx-2], sentence[label_idx-1])
                label_map = {}
                if label in labels_ending:
                    label_map = labels_ending[label]

                label_prev_map = 0
                if prev_label in label_map:
                    label_prev_map = label_map[prev_label]

                label_prev_map += 1
                label_map[prev_label] = label_prev_map
                labels_ending[label] = label_map

    return labels_ending


def build_transition_map_trigram(labels):  # n-gram tagger
    labels_ending = {}

    for sentence in labels:
        for label_idx, label in enumerate(sentence):
            if label_idx > 1:
                prev_label = (sentence[label_idx-2], sentence[label_idx-1])
                label_map = {}
                if label in labels_ending:
                    label_map = labels_ending[label]

                label_prev_map = 0
                if prev_label in label_map:
                    label_prev_map = label_map[prev_label]

                label_prev_map += 1
                label_map[prev_label] = label_prev_map
                labels_ending[label] = label_map

    return labels_ending


def build_transition_map_4gram(labels):  # n-gram tagger
    labels_ending = {}

    for sentence in labels:
        for label_idx, label in enumerate(sentence):
            if label_idx > 2:
                prev_label = (
                    sentence[label_idx-3], sentence[label_idx-2], sentence[label_idx-1])
                label_map = {}
                if label in labels_ending:
                    label_map = labels_ending[label]

                label_prev_map = 0
                if prev_label in label_map:
                    label_prev_map = label_map[prev_label]

                label_prev_map += 1
                label_map[prev_label] = label_prev_map
                labels_ending[label] = label_map

    return labels_ending


def build_emission_map(sentences, labels):
    emission_map = {}

    for sentence, label_set in zip(sentences, labels):
        for word, label in zip(sentence, label_set):
            word_map = {}
            if word in emission_map:
                word_map = emission_map[word]
            if label not in word_map:
                word_map[label] = 0
            word_map[label] += 1
            emission_map[word] = word_map

    return emission_map


def transition(prev_tag, curr_tag, transition_map, smooth_type='add-k', smooth_factor=0.4, prev_tag_prb=0):
    transitions = transition_map[curr_tag]
    denom = 0
    for value in transitions.values():
        denom += value

    if prev_tag not in transitions:
        if smooth_type == 'add-k':
            return smooth_factor / (smooth_factor * len(transition_map))
        if smooth_type == 'linear':
            # Linear Interpolation
            return prev_tag_prb + smooth_factor * (0 - prev_tag_prb)

    num = transitions[prev_tag]

    if smooth_type == 'add-k':
        return (num + smooth_factor) / (denom + smooth_factor * len(transition_map))
    else:
        # Linear Interpolation
        return prev_tag_prb + smooth_factor * (prev_tag_prb - (num / denom))


def emission(word, emission_map, smooth_factor=0):
    if word not in emission_map:
        return []

    word_map = emission_map[word]

    total = 0
    for count in word_map.values():
        total += count

    response = []
    for tag in word_map.keys():
        prb = (word_map[tag] + smooth_factor) / \
            (total + smooth_factor * len(word_map))

        response.append((tag, prb))

    return response


"""
transition_map = build_transition_map_trigram([['A', 'B', 'C','D', 'E'], ['C', 'B', 'B', 'B']])
print(transition_map)
print(transition(('A', 'B'), 'C',transition_map))
emission_map = build_emission_map(
    [['<START>', 'The', 'Fed', 'Did', 'This', '<END>'],
        ['<START>', 'The', 'Pony', 'Did', '<END>']],
    [['<START>', 'A', 'B', 'C', 'A', '<END>'],
        ['<START>', 'C', 'B', 'C', '<END>']
     ]
)
# print(emission_map)
#print(emission('The', emission_map))
def read_array(path):
    with open(path, 'rb') as fp:
        arr = pickle.load(fp)
        return arr
#emission_map = read_array('emission_map.pickle')
#print(emission("The", emission_map, 0.1))

"""
