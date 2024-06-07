
from .segmentation_solution import *
class AbstractSegmenter(object):
    def __init__(self, language_configuration, segmenter_configuration):
        self._language_configuration = language_configuration
        self._segmenter_configuration = segmenter_configuration


    def do_segment(self, signal_data) -> SegmentationSolution:
        raise NotImplementedError
    
