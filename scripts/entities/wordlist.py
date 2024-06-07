import numpy as np
class AbstractEEGWordList(object):
    def __init__(self):
        pass
        
    def deserialize_from(self, file_path):
        raise NotImplementedError
    
    def serialize_to(self, file_path):
        raise NotImplementedError
    
    def append(self, word):
        raise NotImplementedError
    
class NonOrderedEEGWordList(AbstractEEGWordList):
    def __init__(self):
        super().__init__()
    
    def deserialize_from(self, file_path):
        raise NotImplementedError
    
    def serialize_to(self, file_path):
        raise NotImplementedError
    
    def append(self, word):
        raise NotImplementedError
    
class ElectrodeValueRepresentationEEGWordList(AbstractEEGWordList):
    def __init__(self, word_list):
        super().__init__()
        self.word_list = word_list
    
    def deserialize_from(self, file_path):
        self.word_list = np.load(file_path)
        return self
    
    def serialize_to(self, file_path):
        np.save(file_path, self.word_list)
    
    def append(self, word):
        raise NotImplementedError