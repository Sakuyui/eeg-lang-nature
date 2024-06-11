import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")

from typing import List, Set, Tuple
from entities.word import *
from entities.grammar import *

class AbstractEEGGrammar(object):
    def deserialize_from(self, file_path):
        raise NotImplementedError
    def serialize_to(self, file_path):
        raise NotImplementedError
    
class EEGPCFGGrammr(AbstractEEGGrammar):
    def __init__(self, deduct_word: int, deduction_rule: List[int], possibility: int):
        self._deduction_word = deduct_word
        self._deduction_rule = deduction_rule
        self._possibility = possibility
    
    def deserialize_from(self, file_path):
        return super().deserialize_from(file_path)
    
    def serialize_to(self, file_path):
        return super().serialize_to(file_path)
    
    @property
    def possibility(self):
        return self._possibility
    
    @property
    def deduct_word(self):
        return self._deduction_word
    
    @property
    def deduction_rule(self):
        return self._deduction_rule
    
    

class AbstractEEGLanguage(object):
    def __init__(self):
        pass


class EEGPCFGLanguage(AbstractEEGLanguage):
    # (N, T, P, S, R)
    ## N: Nonterminates  [int]
    ## T: Terminates     [int]
    ## S: Start symbol   int
    ## R: Rule set       [N x (N + T)* x P]
    ## P: Possibility set of rules
    def __init__(self, N: Set[int], T: Set[int], S: int, R: List[Tuple[int, List[int]]], P: List[int]):        
        self._N = N
        self._T = T
        self._S = S
        self._R = R
        self._P = P
         
        
    def get_T(self):
        return self._T
    
    def get_S(self):
        return self._S
    
    def get_R(self):
        return self._R
    
    def get_P(self):
        return self._P
    
    def get_N(self):
        return self._N
                
class EEGPCFGLanguageBuild(object):
    def build(self, grammars: List[EEGPCFGGrammr], dictionary: AbstractEEGLanguageDictionary, 
              start_symbol_id: int) -> EEGPCFGLanguage:
        T = set(dictionary.get_word_ids())
        S = start_symbol_id
        P = [grammar.possibility for grammar in grammars]
        R = [(grammar.deduct_word, grammar.deduction_rule) for grammar in grammars]
        N = set([grammar.deduct_word for grammar in grammars])
        return EEGPCFGLanguage(N, T, S, R, P)

