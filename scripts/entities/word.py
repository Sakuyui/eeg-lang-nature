import numpy as np
from typing import List
from .eeg_state import *
class AbstractEEGWord(object):
    def __init__(self):
        pass
    def get_word_id(self):
        raise NotImplemented
    def get_eeg_state_representation_of_word(self):
        raise NotImplemented
    
class StateRepresentedEEGWord(AbstractEEGWord):
    def __init__(self, state: AbstractEEGState, word_id: int):
        self.state = state
        self.tags = {
            'word_id': word_id
        }
    def get_word_id(self):
        return self.tags['word_id']
    def get_eeg_state_representation_of_word(self):
        return self.state
    
class SegmentRepresentedEEGWord(AbstractEEGWord):
    def __init__(self, segment, word_id: int):
        self.segment = segment
        self.tags = {
            'word_id': word_id
        }
    
    def get_word_id(self):
        return self.tags['word_id']
    
    def get_eeg_state_representation_of_word(self):
        return SegmentEEGState(self.segment)
    
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
    
    def get_eeg_state_representation_of_word(self, word_id) -> AbstractEEGState:
        raise NotImplementedError
    
    def get_word_ids(self):
        raise NotImplementedError
    

class DictionaryImplementedEEGLanguageDictionary(AbstractEEGLanguageDictionary):
    def __init__(self, word_list: List[AbstractEEGWord]):
        super().__init__()
        self.dictionary = {word.get_word_id(): word for word in word_list}
    
    def deserialize_from(self, file_path):
        from six.moves import cPickle as pickle
        with open(file_path, 'rb') as f:
            self.dictionary = pickle.load(f)
        return self
    
    def serialize_to(self, file_path):
        from six.moves import cPickle as pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self.dictionary, f)
    
    def append(self, word: AbstractEEGWord) -> AbstractEEGLanguageDictionary:
        self.dictionary[word.get_word_id()] = word
        return self
        
    def get_word_count(self) -> int:
        return len(self.dictionary)
    
    def get_eeg_state_representation_of_word(self, word_id) -> AbstractEEGState:
        word: AbstractEEGWord = self.dictionary[word_id]
        return word.get_eeg_state_representation_of_word()
    
    def get_word_ids(self) -> List:
        return list(self.dictionary.keys())
    