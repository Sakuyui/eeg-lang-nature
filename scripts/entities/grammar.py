import numpy as np
from typing import List, Set
from scripts.entities.word import *
from scripts.entities.grammar import *
class AbstractEEGGrammar(object):
    def deserialize_from(self, file_path):
        raise NotImplementedError
    def serialize_to(self, file_path):
        raise NotImplementedError
    
    
class AbstractEEGLanguage(object):
    def __init__(self):
        pass
    
class EEGPCFGLanguage(AbstractEEGLanguage):
    def __init__(self, grammars: AbstractEEGGrammar = None, dictionary: AbstractEEGLanguageDictionary = None):
        self.grammars : List = []
        self.dictionary : AbstractEEGLanguageDictionary = dictionary
        
    