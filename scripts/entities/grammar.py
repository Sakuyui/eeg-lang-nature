import numpy as np
from typing import List, Set
class AbstractEEGGrammar(object):
    def deserialize_from(self, file_path):
        raise NotImplementedError
    def serialize_to(self, file_path):
        raise NotImplementedError
    
    
class AbstractEEGLanguage(object):
    def __init__(self):
        pass
    
class EEGPCFGLanguage(AbstractEEGLanguage):
    def __init__(self, grammars: List = None, word_list: Set = None):
        self.grammars : List = []
        self.word_list : Set = word_list
        
    