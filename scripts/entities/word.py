import numpy as np
from typing import List
from scripts.entities.eeg_state import *
class AbstractEEGWord(object):
    def __init__(self):
        pass
    def get_word_id(self):
        raise NotImplemented
    
class StateRepresentedEEGWord(AbstractEEGWord):
    def __init__(self, state: AbstractEEGState, word_id: int):
        self.state = state
        self.tags = {
            'word_id': word_id
        }
    def get_word_id(self):
        return self.tags['word_id']
    
class SegmentRepresentedEEGWord(AbstractEEGWord):
    def __init__(self, segment, word_id: int):
        self.segment = segment
        self.tags = {
            'word_id': word_id
        }
    def get_word_id(self):
        return self.tags['word_id']

class AbstractEEGLanguageDictionary(object):
    def __init__(self):
        pass
        
    def deserialize_from(self, file_path):
        raise NotImplementedError
    
    def serialize_to(self, file_path) -> None:
        raise NotImplementedError
    
    def append(self, word):
        raise NotImplementedError
    
    def get_word_count(self) -> int:
        raise NotImplementedError

class DictionaryImplementedEEGLanguageDictionary(AbstractEEGLanguageDictionary):
    def __init__(self, word_list: List[AbstractEEGWord]):
        super().__init__()
        self.dictionary = {word.get_word_id(): word for word in word_list}
    
    def deserialize_from(self, file_path):
        self.dictionary = np.load(file_path, allow_pickle=True)
        return self
    
    def serialize_to(self, file_path):
        np.save(file_path, self.dictionary, allow_pickle=True)
    
    def append(self, word: AbstractEEGWord) -> AbstractEEGLanguageDictionary:
        self.dictionary[word.get_word_id()] = word
        return self
        
    def get_word_count(self) -> int:
        return len(self.dictionary)
    