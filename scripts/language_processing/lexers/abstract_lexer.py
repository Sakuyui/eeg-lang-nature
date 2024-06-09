import sys
sys.path.append("..")

from entities.word import AbstractEEGLanguageDictionary 
from numpy.typing import NDArray
class AbstractLexer(object):
    def segment(self, dictionary: AbstractEEGLanguageDictionary, eeg_data_in_2d_matrix: NDArray):
        raise NotImplementedError