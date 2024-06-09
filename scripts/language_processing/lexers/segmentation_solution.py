class SegmentationSolution(object):
    def __init__(self, word_id_sequence, segment_endpoints):
        if len(segment_endpoints) != len(word_id_sequence) + 1:
            raise ValueError
        self.endpoints = segment_endpoints
        self.word_id_sequence = word_id_sequence
        if len(word_id_sequence) != len(segment_endpoints):
            raise ValueError("the length of word_id_sequence (%d) should equal to the length of endpoints (%d)" % \
                (len(word_id_sequence), len(segment_endpoints)))
            
    
    def get_word_sequence(self):
        return self.word_id_sequence

    def get_segment_endpoints(self):
        return self.endpoints
