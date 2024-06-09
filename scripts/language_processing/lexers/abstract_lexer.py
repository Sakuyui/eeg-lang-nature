import sys
sys.path.append("..")

from entities.word import AbstractEEGLanguageDictionary 
from numpy.typing import NDArray
from .segmentation_solution import *

class AbstractLexer(object):
    def segment(self, dictionary: AbstractEEGLanguageDictionary, eeg_data_in_2d_matrix: NDArray) -> SegmentationSolution:
        raise NotImplementedError