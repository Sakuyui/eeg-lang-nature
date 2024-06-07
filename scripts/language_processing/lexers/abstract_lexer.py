import sys
sys.path.append("..")

from entities.wordlist import * 
from numpy.typing import NDArray
class AbstractLexer(object):
    def segment(self, word_list: AbstractEEGWordList, eeg_data_in_2d_matrix: NDArray):
        raise NotImplementedError