import numpy as np
from numpy.typing import NDArray
from scripts.language_processing.lexers.abstract_lexer import *
from scripts.entities.wordlist import AbstractEEGWordList
import scipy.signal
from utils.topological_graph import eeg2map
import tqdm
from scipy.stats import wasserstein_distance

class GFPElectrodeValueBasedLexer(AbstractLexer) :
    def get_most_similar_word_id(self, word_list: AbstractEEGWordList, word: NDArray, electrode_location_map) -> int:
        word_list_matrix_representation = word_list.word_list
        cnt_word_in_word_list = word_list_matrix_representation.shape[0]
        topological_map_representations = [eeg2map(word_list_matrix_representation[id], electrode_location_map).ravel() for id in range(0, cnt_word_in_word_list)]
        subjective_word_topological_map_representation = eeg2map(word, electrode_location_map).ravel()
        distances = [wasserstein_distance(subjective_word_topological_map_representation, topological_map_representations[id]) for id in range(0, cnt_word_in_word_list)]
        return np.argmax(distances)
        
    def segment(self, word_list: AbstractEEGWordList, eeg_data_in_2d_matrix: NDArray, electrode_location_map) -> NDArray:
        cnt_channels = eeg_data_in_2d_matrix.shape[1]
        electrode_averages = np.average(eeg_data_in_2d_matrix, axis=0)
        gfps = np.sqrt(np.sum((eeg_data_in_2d_matrix - electrode_averages) ** 2 / cnt_channels, axis=0))
        gfp_peaks = scipy.signal.find_peaks(gfps)[0]
        
        word_sequence = np.zeros((len(gfp_peaks)))
        peak_index = 0
        for peak_index in tqdm.tqdm(range(0, len(gfp_peaks))):
            peak = gfp_peaks[peak_index]
            state_t = eeg_data_in_2d_matrix[:, peak]
            word_sequence[peak_index] = self.get_most_similar_word_id(word_list, state_t, electrode_location_map)
        
        return word_sequence