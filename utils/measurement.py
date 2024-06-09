from scripts.entities.eeg_state import *
from scipy.stats import wasserstein_distance
from typing import Dict

def eeg_state_distance_topological(eeg_state1: AbstractEEGState, eeg_state2: AbstractEEGState, electrode_location_configuration: Dict):
        eeg_state_1_topological_map = eeg_state1.to_topological_representation(electrode_location_configuration)
        eeg_state_2_topological_map = eeg_state2.to_topological_representation(electrode_location_configuration)
        return wasserstein_distance(eeg_state_1_topological_map.ravel(), eeg_state_2_topological_map.ravel())
    
