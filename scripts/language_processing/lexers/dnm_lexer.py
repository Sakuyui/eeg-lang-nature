from .segmentation_solution import *
from .abstract_lexer import *
import numpy as np
import scipy

class DNMLexer(AbstractLexer):
    def __init__(self, language_configuration):
        super().__init__()
        self._word_list_size = language_configuration.configuration_word_count()

    def _calculate_DNM_indexes(self, signal_data):
        cnt_time_slots = signal_data.shape[1]
        n_channels = signal_data.shape[0]

        DNM_indexes = np.zeros((cnt_time_slots))
        DNM_index = 0.0
        SD_factor = 0.0
        correlation_factor = 0.0
        cnt_pairs = 0
        for t in range(cnt_time_slots):
            cnt_pairs = 0
            avg = np.average(signal_data[:, t])
            SD_factor = (np.sum((signal_data[:, t] - avg) ** 2)) ** (1 / n_channels)
            for e_from in range(n_channels):
                for e_to in range(e_from + 1, n_channels):
                    correlation_factor += np.corrcoef(signal_data[e_from, t], signal_data[e_to, t])[0, 1] ** 2
                    cnt_pairs += 1
            DNM_index = SD_factor * np.sqrt(correlation_factor)
            DNM_indexes[t] = DNM_index
            
        return DNM_indexes
    
    def segment(self, dictionary: AbstractEEGLanguageDictionary, eeg_data_in_2d_matrix: NDArray):
        DNM_indexes = self._calculate_DNM_indexes(eeg_data_in_2d_matrix)
        peaks = scipy.signal.find_peaks(DNM_indexes)[0]
        return peaks