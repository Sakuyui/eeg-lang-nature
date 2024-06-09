from typing import Dict
from numpy.typing import NDArray
from utils.topological_graph import eeg2map, eeg_to_topological_graph_mds
class AbstractEEGState(object):
    def __init__(self):
        pass
    def to_topological_representation(self, eleteode_location_configuration):
        raise NotImplementedError
    
class TopologicalMapEEGState(AbstractEEGState):
    def __init__(self, topological_map, eletrode_location_configuration):
        self.topological_map = topological_map
        self.eletrode_location_configuration = eletrode_location_configuration
    def to_topological_representation(self, eleteode_location_configuration):
        return self.topological_map
        
class SegmentEEGState(AbstractEEGState):
    def __init__(self, segment):
        self.segment = segment
    def to_topological_representation(self, eleteode_location_configuration):
        raise NotImplementedError
    
class EletrodeValuesEEGState(AbstractEEGState):
    def __init__(self, eleteode_values: NDArray):
        self.eleteode_values = eleteode_values
    
    def to_topological_representation(self, eleteode_location_configuration) -> TopologicalMapEEGState:
        return eeg_to_topological_graph_mds(self.eleteode_values, eleteode_location_configuration)
    
    
