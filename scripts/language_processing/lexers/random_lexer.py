import numpy as np
from numpy.typing import NDArray

from scripts.entities.word import AbstractEEGLanguageDictionary
from .segmentation_solution import *
from lexers.abstract_lexer import *

class RandomLexer(AbstractLexer):
    def __init__(self, max_segment_length, dictionary_size):
        super().__init__()
        self._max_segment_length = max_segment_length
        self._dictionary_size = dictionary_size
    
    def segment(self, dictionary: AbstractEEGLanguageDictionary, eeg_data_in_2d_matrix: np.ndarray[np.Any, np.dtype]):
        print("segmenting signal data with shape of %s" % str(eeg_data_in_2d_matrix.shape))
        cnt_time_slots = eeg_data_in_2d_matrix.shape[1]
        segment_endpoints = [0]
        word_id_sequence = []
        pos = 0
        while pos < cnt_time_slots:
            random_length = np.random.randint(0, self._max_segment_length)
            pos += random_length
            pos = min(cnt_time_slots, pos)
            segment_endpoints.append(pos)
            word_id_sequence.append(np.random.randint(0, self._dictionary_size))
        return SegmentationSolution(word_id_sequence, segment_endpoints)