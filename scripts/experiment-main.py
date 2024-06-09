# Pasre the EEG Data
from typing import Dict
from entities.grammar import *
from entities.word import *
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
    
    def is_configuration_exist(self, configure_name, recursively = False) -> bool:
        def recursively_check(dictionary):
            if dictionary == None or not isinstance(dictionary, dict):
                return False
            if configure_name in dictionary:
                return True
            for key in dictionary.keys():
                check_result = recursively_check(dictionary[key])
                if check_result:
                    return True
            return False
            
        if configure_name in self.configuration:
            return True
        if not recursively:
            return False
        return recursively_check(self.configuration)
    
    def get_bool_congiguration_item(self, item_name, recursively = False):
        result = self.get_configuration_item_if_exist(item_name, recursively)
        if result == None or not isinstance(result, bool):
            return False
        return result
    
    def get_configuration_item_if_exist(self, item_name, recursively = False):
        def recursively_get(dictionary):
            if dictionary == None or not isinstance(dictionary, dict):
                return None
            if item_name in dictionary:
                return dictionary[item_name]
            for key in dictionary.keys():
                result = recursively_get(dictionary[key])
                if result != None:
                    return result
            return None
            
        if item_name in self.configuration:
            return self.configuration[item_name]
        if not recursively:
            return None
        return recursively_get(self.configuration)
        
    def load_eeg_data(self):
        raw_file_path = self.get_configuration_item_if_exist('raw_file_path')
        print("Load raw file from %s" % raw_file_path)
        raw_data = mne.io.read_raw(raw_file_path)
        info = raw_data.info
        if self.is_configuration_exist("ch_names"):
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
    
    def select_dictionary_builder(self) -> AbstractWordListBuilder:
        dictionary_builder_name = self.configuration['dictionary_builder']
        print("  > Choose word list builder = %s" %  dictionary_builder_name)
        if 'GFP_Kmeans' == dictionary_builder_name:
            return GFPKmeansEEGLanguageDictionaryBuilder(self.electrode_location_map)
        if 'dummy' == dictionary_builder_name:
            return DummyEEGLanguageDictionaryBuilder()
        if 'random' == dictionary_builder_name:
            return 
    
    def build_dictionary(self, eeg_matrix, eeg_info) -> AbstractEEGLanguageDictionary:
        dictionary_builder: AbstractWordListBuilder = self.select_dictionary_builder()
        dictionary = None
        
        if self.get_bool_congiguration_item('build_dictionary_from_file'):
            dictionary = dictionary_builder.deserialize_from_file(self.configuration['input_dictionary_file_path'])
            print(" ------ deserialize word list from %s ------" % self.configuration['input_dictionary_file_path'])
            return dictionary
        dictionary = dictionary_builder.build_dictionary(eeg_matrix, eeg_info, self.electrode_location_map)
        return dictionary
    
    def select_grammar_extractor(self) -> AbstracrGrammarExtractor:
        raise NotImplementedError
    
    def grammar_extraction(self, eeg_matrix, eeg_info, dictionary) -> AbstractEEGGrammar:
        grammar_extractor: AbstracrGrammarExtractor =  self.select_grammar_extractor(self.configuration)
        return grammar_extractor.extract_grammar(eeg_matrix, eeg_info, dictionary)
    
    def select_parser(self) -> AbstractEEGParser:
        raise NotImplementedError
    
    def select_model(self) -> AbstractPreEpilepticStatePredictionModel:
        raise NotImplementedError
    
    def build_model(self) -> AbstractPreEpilepticStatePredictionModel:
        model_builder = AbstractModelBuilder()
        model = model_builder.build_model(self.select_model(), self.configuration)
        raise NotImplementedError
    
    def evaluation_model(self, eeg_matrix, eeg_info, dictionary, grammar, parser, prediction_model):
        raise NotImplementedError
    
    def _segment(self, dataset_for_segmentation, dictionary):
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
        dictionary: AbstractEEGLanguageDictionary = self.build_dictionary(eeg_matrix, eeg_info)
        if self.get_bool_congiguration_item('serialize_dictionary'):
            print('  ------ Serialize Word List to %s ------' % self.configuration['output_dictionary_file_path'])
            dictionary.serialize_to(self.configuration['output_dictionary_file_path'])
        
        # segmentation
        if dataset_for_segmentation == None:
            dataset_for_segmentation = eeg_matrix
        
        print("  ------ Segmenting ------")
        word_sequence = None
        if self.get_bool_congiguration_item("load_word_sequence_from_file"):
            word_sequence = np.load(self.configuration['input_word_sequence_file_path'])
        else:
            word_sequence = self.select_lexer().segment(dictionary, dataset_for_segmentation, self.electrode_location_map)
        
        print(word_sequence)
        if not self.get_bool_congiguration_item('build_dictionary_from_file') and self.get_bool_congiguration_item('save_word_sequence'):
            print('  ------ Save Segmentation Results (Word Sequence) to %s ------' % self.configuration['output_word_sequence_file_path'])
            np.save(self.configuration['output_word_sequence_file_path'], word_sequence)
        print("-------- End Building EEG Language --------")

    def evaluation_experiment(self):
        # Load EEG data and its information according to the configuration given when calling this method.
        eeg_data = self.load_eeg_data(self.configuration)
        eeg_matrix = self.get_eeg_data_matrix(eeg_data)
        eeg_info = self.get_eeg_info(eeg_data)
        # Extract Word List
        dictionary: AbstractEEGLanguageDictionary = self.build_dictionary(eeg_matrix, eeg_info)
        # Grammar Extraction
        grammar: AbstractEEGGrammar = self.grammar_extraction(eeg_matrix, eeg_info, dictionary)
        # Select EEG Data Parser
        parser: AbstractEEGParser = self.select_parser(self.configuration)
        # Model building
        prediction_model: AbstractPreEpilepticStatePredictionModel = self.build_model(self.configuration)
        # Evaluation
        self.evaluate(eeg_matrix, eeg_info, dictionary, grammar, parser, prediction_model, self.configuration)
        
    def model_trainning(self):
        # load EEG data and its information according to the configuration given when calling this method.
        eeg_data = self.load_eeg_data(self.configuration)
        eeg_matrix = self.get_eeg_data_matrix(eeg_data)
        eeg_info = self.get_eeg_info(eeg_data)
        # Extract Word List
        dictionary: AbstractEEGLanguageDictionary = self.build_dictionary(eeg_matrix, eeg_info)
        # Grammar Extraction
        grammar: AbstractEEGGrammar = self.grammar_extraction(eeg_matrix, eeg_info, dictionary)
        # Select EEG Data Parser
        parser: AbstractEEGParser = self.select_parser(self.configuration)
        # Model building
        prediction_model: AbstractPreEpilepticStatePredictionModel = self.build_model(self.configuration)
        # Evaluation
        prediction_model.training(eeg_matrix, eeg_info, dictionary, grammar, parser)

experiment_configuration1 = {
    'raw_file_path': '../data/dataset1/Raw_EDF_Files/p10_Record1.edf',
    'ch_names': ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','A1','A2'],
    'montage': 'standard_1020',
    'dictionary_builder': 'dummy',
    'dictionary_builder_configuration':{
        'cnt_word': 10  
    },
    'serialize_dictionary': True,
    'output_dictionary_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/dictionary.wl.npy',
    'build_dictionary_from_file': True,
    'input_dictionary_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/dictionary.wl.npy',
    'lexer': 'gfp-electrode-value-based-lexer',
    'save_word_sequence': True,
    'load_word_sequence_from_file': True,
    'input_word_sequence_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/word_sequence.ws.npy',
    'output_word_sequence_file_path': 'C:/Users/Micro/Desktop/Research/eeg-language/data/word_sequence.ws.npy',

}

experiment = Experiment(experiment_configuration1)
experiment.build_language()