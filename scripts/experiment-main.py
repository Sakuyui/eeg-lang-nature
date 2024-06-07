# Pasre the EEG Data
# eeg_syntax_tree: AbstractEEGSyntaxTree = parser.parse(eeg_info, eeg_matrix, word_list, grammar)
from typing import Dict
from entities.grammar import *
from entities.wordlist import *
from evaluators.abstract_evaluator import *
from applications.prediction_models.abstract_prediction_model import *
from applications.prediction_models.abstract_segment_preprocessing_model import *
from applications.prediction_models.model_builder import *
from language_processing.parsers.parser import *
from language_processing.language_builder.language_builder import *
from language_processing.language_builder.configuration import *
import mne

class Experiment(object):
    def __init__(self, configuration: Dict):
        self.configuration = configuration
    
    def load_eeg_data(self):
        raw_file_path = self.configuration['raw_file_path']
        print("Load raw file from %s" % raw_file_path)
        raw_data = mne.io.read_raw(raw_file_path)
        info = raw_data.info
        if "ch_names" in self.configuration:
            mne.rename_channels(info, {x:y for x, y in zip(info.ch_names, self.configuration['ch_names'])})
        montage = mne.channels.make_standard_montage(self.configuration['montage'])
        raw_data.set_montage(montage)
        self.loaded_raw_data = raw_data
        self.electrode_location_map = [(ch_name, dig['r']) for dig, ch_name in list(zip(montage.dig, montage.ch_names)) if ch_name in self.configuration['ch_names']]
        return raw_data
    
    def preprocess(self, eeg_matrix):
        return eeg_matrix
    
    def get_eeg_data_matrix(self, eeg_data):
        return eeg_data.get_data()
    
    def get_eeg_info(self, eeg_data):
        return eeg_data.info
    
    def select_word_list_builder(self) -> AbstractWordListBuilder:
        word_list_builder_name = self.configuration['word_list_builder']
        print("  > Choose word list builder = %s" %  word_list_builder_name)
        if 'GFP_Kmeans' == word_list_builder_name:
            return GFPKmeansWordListBuilder(self.electrode_location_map)
    
    def word_list_building(self, eeg_matrix, eeg_info) -> AbstractEEGWordList:
        word_list_builder: AbstractWordListBuilder = self.select_word_list_builder()
        word_list = None
        if 'build_word_list_from_file' in self.configuration and self.configuration['build_word_list_from_file']:
            word_list = word_list_builder.deserialize_from_file(self.configuration['input_word_list_file_path'])
            print(" ------ deserialize word list from %s ------" % self.configuration['input_word_list_file_path'])
            return word_list
        word_list = word_list_builder.build_word_list(eeg_matrix, eeg_info, self.electrode_location_map)
        return word_list
    
    def select_grammar_extractor(self) -> AbstracrGrammarExtractor:
        raise NotImplementedError
    
    def grammar_extraction(self, eeg_matrix, eeg_info, word_list) -> AbstractEEGGrammar:
        grammar_extractor: AbstracrGrammarExtractor =  self.select_grammar_extractor(self.configuration)
        return grammar_extractor.extract_grammar(eeg_matrix, eeg_info, word_list)
    
    def select_parser(self) -> AbstractEEGParser:
        raise NotImplementedError
    
    def select_model(self) -> AbstractPreEpilepticStatePredictionModel:
        raise NotImplementedError
    
    def build_model(self) -> AbstractPreEpilepticStatePredictionModel:
        model_builder = AbstractModelBuilder()
        model = model_builder.build_model(self.select_model(), self.configuration)
        raise NotImplementedError
    
    def evaluation_model(self, eeg_matrix, eeg_info, word_list, grammar, parser, prediction_model):
        raise NotImplementedError
    
    def _segment(self, dataset_for_segmentation, word_list):
        word_sequence = []
    
    def select_lexer(self):
        lexer = None
        if 'gfp-electrode-value-based-lexer' == self.configuration['lexer']:
            from language_processing.lexers.gfp_lexers import GFPElectrodeValueBasedLexer
            lexer = GFPElectrodeValueBasedLexer()
        return lexer
    
    def build_language(self, dataset_for_segmentation = None):
        print("-------- Begin Building EEG Language --------")
        eeg_data = self.load_eeg_data()
        eeg_matrix = self.get_eeg_data_matrix(eeg_data)
        eeg_info = self.get_eeg_info(eeg_data)
        eeg_matrix = self.preprocess(eeg_matrix)
        word_list: AbstractEEGWordList = self.word_list_building(eeg_matrix, eeg_info)
        if 'serialize_word_list' in self.configuration and self.configuration['serialize_word_list']:
            print('  ------ Serialize Word List to %s ------' % self.configuration['output_word_list_file_path'])
            word_list.serialize_to(self.configuration['output_word_list_file_path'])
        
        # segmentation
        if dataset_for_segmentation == None:
            dataset_for_segmentation = eeg_matrix
        
        print("  ------ Segmenting ------")
        word_sequence = self.select_lexer().segment(word_list, dataset_for_segmentation, self.electrode_location_map)
        print(word_sequence)
        if 'save_word_sequence' in self.configuration and self.configuration['save_word_sequence']:
            print('  ------ Save Word Sequence to %s ------' % self.configuration['output_word_list_file_path'])
            np.save(self.configuration['word_sequence_file_path'], word_sequence)
        print("-------- End Building EEG Language --------")

    def evaluation_experiment(self):
        # Load EEG data and its information according to the configuration given when calling this method.
        eeg_data = self.load_eeg_data(self.configuration)
        eeg_matrix = self.get_eeg_data_matrix(eeg_data)
        eeg_info = self.get_eeg_info(eeg_data)
        # Extract Word List
        word_list: AbstractEEGWordList = self.word_list_building(eeg_matrix, eeg_info)
        # Grammar Extraction
        grammar: AbstractEEGGrammar = self.grammar_extraction(eeg_matrix, eeg_info, word_list)
        # Select EEG Data Parser
        parser: AbstractEEGParser = self.select_parser(self.configuration)
        # Model building
        prediction_model: AbstractPreEpilepticStatePredictionModel = self.build_model(self.configuration)
        # Evaluation
        self.evaluation(eeg_matrix, eeg_info, word_list, grammar, parser, prediction_model, self.configuration)
        
    def model_trainning(self):
        # load EEG data and its information according to the configuration given when calling this method.
        eeg_data = self.load_eeg_data(self.configuration)
        eeg_matrix = self.get_eeg_data_matrix(eeg_data)
        eeg_info = self.get_eeg_info(eeg_data)
        # Extract Word List
        word_list: AbstractEEGWordList = self.word_list_building(eeg_matrix, eeg_info)
        # Grammar Extraction
        grammar: AbstractEEGGrammar = self.grammar_extraction(eeg_matrix, eeg_info, word_list)
        # Select EEG Data Parser
        parser: AbstractEEGParser = self.select_parser(self.configuration)
        # Model building
        prediction_model: AbstractPreEpilepticStatePredictionModel = self.build_model(self.configuration)
        # Evaluation
        prediction_model.training(eeg_matrix, eeg_info, word_list, grammar, parser)

experiment_configuration1 = {
    'raw_file_path': '../data/dataset1/Raw_EDF_Files/p10_Record1.edf',
    'ch_names': ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','A1','A2'],
    'montage': 'standard_1020',
    'word_list_builder': 'GFP_Kmeans',
    'serialize_word_list': True,
    'output_word_list_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/word_list.wl.npy',
    'build_word_list_from_file': True,
    'input_word_list_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/word_list.wl.npy',
    'lexer': 'gfp-electrode-value-based-lexer',
    'save_word_sequence': True,
    'word_sequence_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/word_sequence.ws.npy'
}

experiment = Experiment(experiment_configuration1)
experiment.build_language()