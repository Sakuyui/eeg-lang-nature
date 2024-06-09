import numpy as np
from numpy.typing import NDArray
from scripts.language_processing.lexers.abstract_lexer import *
from scripts.entities.word import AbstractEEGLanguageDictionary
import scipy.signal
from utils.topological_graph import eeg2map
from utils.measurement import eeg_state_distance_topological
import tqdm
from scipy.stats import wasserstein_distance
from scripts.entities.eeg_state import EletrodeValuesEEGState, AbstractEEGState


class GFPElectrodeValueBasedLexer(AbstractLexer) :
    def get_most_similar_word_id(self, eeg_language_dictionary: AbstractEEGLanguageDictionary, state: AbstractEEGState, electrode_location_configuration) -> int:
        cnt_word_in_dictionary = eeg_language_dictionary.get_word_count()
        distances = [eeg_state_distance_topological(eeg_language_dictionary.get_eeg_state_representation_of_word(id), state, electrode_location_configuration) for id in range(0, cnt_word_in_dictionary)]
        return np.argmax(distances)
        
    def segment(self, dictionary: AbstractEEGLanguageDictionary, eeg_data: NDArray, electrode_location_configuration) -> NDArray:
        cnt_channels = eeg_data.shape[0]
        electrode_averages = np.average(eeg_data, axis=0)
        gfps = np.sqrt(np.sum((eeg_data - electrode_averages) ** 2 / cnt_channels, axis=0))
        gfp_peaks = scipy.signal.find_peaks(gfps)[0]
        
        word_sequence = np.zeros((len(gfp_peaks)))
        peak_index = 0
        for peak_index in tqdm.tqdm(range(0, len(gfp_peaks))):
            peak = gfp_peaks[peak_index]
            state_t = eeg_data[:, peak]
            word_sequence[peak_index] = self.get_most_similar_word_id(dictionary, EletrodeValuesEEGState(state_t), electrode_location_configuration)
        
        return SegmentationSolution(word_sequence, gfp_peaks)