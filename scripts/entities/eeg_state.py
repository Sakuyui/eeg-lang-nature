from typing import Dict
from numpy.typing import NDArray
class AbstractEEGState(object):
    def __init__(self):
        pass
    def to_topological_representation(self):
        raise NotImplementedError
    
class TopologicalMapEEGState(AbstractEEGState):
    def __init__(self, topological_map, eletrode_location_configuration):
        self.topological_map = topological_map

class EletrodeValuesEEGState(AbstractEEGState):
    def __init__(self, eleteode_values: NDArray):
        self.eleteode_values = eleteode_values
    
    def to_topological_representation(self) -> TopologicalMapEEGState:
        raise NotImplementedError

