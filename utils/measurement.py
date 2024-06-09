from scripts.entities.eeg_state import *
from scipy.stats import wasserstein_distance
from typing import Dict

def eeg_state_distance_topological(eeg_state1: AbstractEEGState, eeg_state2: AbstractEEGState, electrode_location_configuration: Dict):
        if not isinstance(eeg_state1, TopologicalMapEEGState):
            eeg_state1 = eeg_state1.to_topological_representation()
        if not isinstance(eeg_state2, TopologicalMapEEGState):
            eeg_state2 = eeg_state1.to_topological_representation()
        return wasserstein_distance(eeg_state1.ravel(), eeg_state2.ravel())
    
