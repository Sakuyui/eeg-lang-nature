from typing import List
from ga_solution import AbstractGASolution

class AbstractGeneticAlgorithmSolutionGenerator(object):
    def generate_init_solutions(size, cnt_word: int) -> List[AbstractGASolution]:
        raise NotImplementedError
    
class RandomGeneticAlgorithmSolutionGenerator(AbstractGeneticAlgorithmSolutionGenerator):
    def generate_init_solutions(size, cnt_word: int):
        solutions = []
    