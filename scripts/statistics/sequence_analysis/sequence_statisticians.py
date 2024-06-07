import numpy as np
import sys
sys.path.append("..")
from microstate_stat_helper import testMarkov0, testMarkov1, testMarkov2

def calculate_entropy(sequence):
    element_counting_record = {}
    for element in sequence:
        if element not in element_counting_record:
            element_counting_record[element] = 0
        element_counting_record[element] += 1
    counts = list(element_counting_record.values())
    freqs = counts / np.sum(counts)
    del counts
    entropy = [freq * np.log(freq) for freq in freqs]
    return entropy

def calculate_max_entropy(element_count: int, base = 2):
    p = 1.0 / element_count # entropy reach maximum when the distribution of element obey uniform distribution
    return -element_count * (np.log(element_count) / np.log(base))

        
def calculate_empirical_entropy(sequence, base = 2):
    empirical_symbol_distribution = calculate_empirical_symbol_distribution(sequence, base)
    entropy = 0
    for i in range(len(empirical_symbol_distribution)):
        entropy += empirical_symbol_distribution[i] * (np.log(empirical_symbol_distribution[i]) / np.log(base))
    return -entropy

def calculate_empirical_symbol_distribution(sequence, cnt_element_categories):
        mat = np.zeros((cnt_element_categories))
        pos = 0
        while pos < len(sequence):
            element = sequence[pos]
            mat[element] += 1
            pos += 1
        return (mat / np.sum(mat)).reshape((cnt_element_categories, 1))
    
def calculate_transition_matrix(sequence, cnt_element_categories):
        mat = np.zeros((cnt_element_categories, cnt_element_categories))
        pos = 1
        while pos < len(sequence):
            previous_word_id = sequence[pos - 1]
            current_word_id = sequence[pos]
            mat[previous_word_id, current_word_id] += 1
            pos += 1
        mat /= np.sum(mat, axis = 1)[:, np.newaxis]
        return mat
    
def markov_tests(sequence, cnt_element_categories, order=0, alpha=0.01):
        if order == 0:
            testMarkov0(sequence, cnt_element_categories, alpha)
        elif order == 1:
            testMarkov1(sequence, cnt_element_categories, alpha)
        elif order == 2:
            testMarkov2(sequence, cnt_element_categories, alpha)
        else:
            raise NotImplementedError