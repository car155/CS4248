# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import datetime
from collections import defaultdict
import pickle
import numpy as np
import re

# tags
START = "<s>"
END = "</s>"
UNK = "<UNK>"
unk_threshold = 1

# suffixes
SUFFIXES = [r".*ion$", r".*ive$", r".*ing$", r".*est$", r".*al$", r".*ly$", r".*en$", r".*ed$", r".*er$", r".*es$", r".*s$", r".*-", r".*\."]
def get_suffix(word):
    for i in range(len(SUFFIXES)):
        if re.match(SUFFIXES[i], word):
            return i+1
    return 0

# capitalisation
def get_capitalisation(word):
    if (word.islower()): # all lower
        return 0
    elif (word.isupper()): # all upper
        return 1
    elif (word[0].isupper()): # first upper
        return 2
    else:
        return 3 # any upper

# witten bell smoothing for transition probabilities
def compute_transition_probs_witten_bell(index, count, bigram_pairs, bigram_count):
    transition_probs = np.zeros((len(index), len(index)))

    for w_0 in index:
        T = len(bigram_pairs[w_0])
        Z = len(index) - T
        for w in index:
            if bigram_count[(w_0, w)] > 0:
                transition_probs[index[w_0], index[w]] = bigram_count[(w_0, w)] / (count[w_0] + T)
            else:
                transition_probs[index[w_0], index[w]] = T / (Z * (count[w_0] + T))

    return transition_probs

def compute_emission_probs(pos_index, pos_count, word_index, emission_count):
    emission_probs = np.zeros((len(pos_index), len(word_index)))

    for (pos, word), count in emission_count.items():
        emission_probs[pos_index[pos], word_index[word]] = count / pos_count[pos]

    return emission_probs

def compute_capital_probs(pos_index, pos_count, capital_count):
    capital_probs = np.zeros((len(pos_index), 4))

    for (pos, capital), count in capital_count.items():
        capital_probs[pos_index[pos], capital] = count / pos_count[pos]

    return capital_probs

def compute_suffix_probs(pos_index, pos_count, suffix_count):
    suffix_probs = np.zeros((len(pos_index), len(SUFFIXES)+1))

    for (pos, suffix), count in suffix_count.items():
        suffix_probs[pos_index[pos], suffix] = count / pos_count[pos]

    return suffix_probs

def train_model(train_file, model_file):
    # defaultdict(int) -> 0 if not previously keyed
    # defaultdict(set) -> empty set if not previously keyed
    pos_count = defaultdict(int) # c(t_i)
    word_count = defaultdict(int) # c(w_i)
    # hmm
    transition_count = defaultdict(int) # c(t_{i-1}, t_i)
    transition_pairs = defaultdict(set) # for key t_{i-1} what t_i can it go to
    emission_count = defaultdict(int) # c(w_i, t_i)
    emission_pairs = defaultdict(set) # for key t_i what w_i can happen
    # unknown handling
    capital_count = defaultdict(int)  
    suffix_count = defaultdict(int)

    with open(train_file) as f:
        lines = f.readlines()
        # Initialise word_count
        for line in lines:
            for word_pos_pairs in line.split():
                word_pos_pairs = word_pos_pairs.rsplit('/', 1)
                word, pos = word_pos_pairs[0], word_pos_pairs[1]
                word_count[word] += 1

        for line in lines:
            # start tag
            prev_pos = START
            pos_count[START] += 1
            
            for word_pos_pairs in line.split():
                word_pos_pairs = word_pos_pairs.rsplit('/',1)
                word, pos = word_pos_pairs[0], word_pos_pairs[1]

                # for unk estimation
                suffix_count[(pos, get_suffix(word))] += 1
                capital_count[(pos, get_capitalisation(word))] += 1

                # Unknown estimation smoothing for word emission probabilities
                if (word_count[word] <= unk_threshold):
                    word_count.pop(word)
                    word = UNK
                    word_count[UNK] += 1

                # p(t_i | t_{i-1})
                transition_count[(prev_pos, pos)] += 1
                pos_count[pos] += 1
                transition_pairs[prev_pos].add(pos)

                # p(w_i | t_i)
                emission_count[(pos, word)] += 1
                emission_pairs[pos].add(word)

                prev_pos = pos

        # end tag
        transition_count[(prev_pos, END)] += 1
        transition_pairs[prev_pos].add(END)
        pos_count[END] += 1

        pos_index = {} # mapping from pos to index in transition prob grid
        pos_list = {} # mapping from index in transition prob grid to pos
        word_index = {} # mapping from word to index in emission prob grid
        word_list = {} # mapping from index in emission prob grid to word

        for i, pos in enumerate(pos_count.keys()):
            pos_index[pos] = i
            pos_list[i] = pos
        for i, word in enumerate(word_count.keys()):
            word_index[word] = i
            word_list[i] = word

        transition_probs = compute_transition_probs_witten_bell(pos_index, pos_count, transition_pairs, transition_count)
        emission_probs = compute_emission_probs(pos_index, pos_count, word_index, emission_count)
        capital_probs = compute_capital_probs(pos_index, pos_count, capital_count)
        suffix_probs = compute_suffix_probs(pos_index, pos_count, suffix_count)

    with open(model_file, 'wb') as f:
        pickle.dump({
            'word_index': word_index,
            'pos_index': pos_index,
            'pos_list': pos_list,
            'transition_probs': transition_probs,
            'emission_probs': emission_probs,
            'suffix_probs': suffix_probs,
            'capital_probs': capital_probs,
        }, f)

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)