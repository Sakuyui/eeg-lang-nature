from typing import List
from scripts.entities import AbstractEEGLanguage

class AbstractGASolution(object):
    def __init__(self, solution_representation) -> None:
        self.solution_representation = solution_representation
    
    @property
    def data(self):
        return self.solution_representation

class BytearrayRepresentedGASolution(AbstractGASolution):
    def __init__(self, solution_representation: bytearray) -> None:
        self.solution_representation = solution_representation

class IntArrayRepresentedGASolution(AbstractGASolution):
    def __init__(self, solution_representation: List[int]) -> None:
        self.solution_representation = solution_representation

class EEGLanguageRepresentedGASolution(AbstractGASolution):
    def __init__(self, solution_representation: AbstractEEGLanguage) -> None:
        self.solution_representation = solution_representation


