import numpy as np
from .segmentation_solution import *
from .abstract_segmentor import *

class RandomSegmenter(AbstractSegmenter):
    def __init__(self, language_configuration, segmenter_configuration):
        super().__init__(language_configuration, segmenter_configuration)
        self._word_list_size = language_configuration.configuration_word_count()
        self._max_segment_length = self._segmenter_configuration['max_segment_size']

    def do_segment(self, signal_data) -> SegmentationSolution:
        print("segmenting signal data with shape of %s" % str(signal_data.shape))
        cnt_time_slots = signal_data.shape[1]
        segment_endpoints = [0]
        word_id_sequence = []
        pos = 0
        while pos < cnt_time_slots:
            random_length = np.random.randint(0, self._max_segment_length)
            pos += random_length
            pos = min(cnt_time_slots, pos)
            segment_endpoints.append(pos)
            word_id_sequence.append(np.random.randint(0, self._word_list_size))
        return SegmentationSolution(word_id_sequence, segment_endpoints)
            