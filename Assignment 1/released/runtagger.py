# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys
import datetime
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

def process_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        pos_index = model['pos_index']
        pos_list = model['pos_list']
        word_index = model['word_index']
        transition_probs = model['transition_probs']
        emission_probs = model['emission_probs']
        capital_probs = model['capital_probs']
        suffix_probs = model['suffix_probs']
    
    return pos_index, pos_list, word_index, transition_probs, emission_probs, capital_probs, suffix_probs

def convert_to_log(p):
    p_log = np.log(p, where=p!=0, dtype=float)
    p_log[p==0] = float("-inf")
    return p_log
 
def compute_viterbi(line, pos_index, word_index, transition_probs, emission_probs, capital_probs, suffix_probs):
    words = line.split()
    N, T = len(pos_index), len(words)
    v = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # convert to logs for easier multiplication
    transition_probs = convert_to_log(transition_probs)
    emission_probs = convert_to_log(emission_probs)
    capital_probs = convert_to_log(capital_probs)
    suffix_probs = convert_to_log(suffix_probs)

    # init
    w = words[0]
    if w in word_index:
        b = emission_probs[:, word_index[w]]
    else:
        # p(w_i|t_i) = p(unk|t_i) * p(capital|t_i) * p(suffix|t_i)
        b = emission_probs[:, word_index[UNK]]
        suffix = get_suffix(w)
        capital = get_capitalisation(w)
        b = b + suffix_probs[:, suffix] + capital_probs[:, capital]
    
    # v[s, 0] = a_(0,s) * b_s(o_0)
    a = transition_probs[pos_index[START], :]
    v[:, 0] = a + b
    # backpointer[s,1] -> start
    backpointer[:, 0] = pos_index[START]

    for t in range(1, T):
        for s in range(N):
            w = words[t] # o_t
            prev_v = v[:, t-1] # v[s',t-1]            
            a = transition_probs[:, s] # a[s',s]
            if w in word_index:
                b = emission_probs[s, word_index[w]] # b_s(o_t)
            else:
                b = emission_probs[s, word_index[UNK]]
                suffix = get_suffix(w)
                capital = get_capitalisation(w)
                b = b + suffix_probs[s, suffix] + capital_probs[s, capital]
            s_prev = np.argmax(prev_v + a)
            backpointer[s, t] = s_prev
            v[s, t] = prev_v[s_prev] + a[s_prev] + b
    
    s_prev = np.argmax(v[:, T-1] + transition_probs[:, pos_index[END]])
    backpointer[pos_index[END], T-1] = s_prev
    v[pos_index[END], T-1] = v[s_prev, T-1] + transition_probs[s_prev, pos_index[END]]

    return backpointer

def backtrace(backpointer, line, pos_index, pos_list):
    words = line.split()
    T = len(words)
    output = "\n"
    current = backpointer[pos_index[END], T-1]
    for t in range(T-1, 0, -1):
        output = " " + words[t] + "/" + pos_list[current] + output
        current = backpointer[current, t] 
    output = words[0] + "/" + pos_list[current] + output

    return output 

def tag_sentence(test_file, model_file, out_file):
    pos_index, pos_list, word_index, transition_probs, emission_probs, capital_probs, suffix_probs = process_model(model_file)

    with open(test_file) as f:
        with open(out_file, 'w+') as f_output:
            lines = f.readlines()
            counter = 0
            for line in lines:
                # print((counter / len(lines)) * 100)
                backpointer = compute_viterbi(line, pos_index, word_index, transition_probs, emission_probs, capital_probs, suffix_probs)
                output_str = backtrace(backpointer, line, pos_index, pos_list)
                f_output.write(output_str)
                counter += 1

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)