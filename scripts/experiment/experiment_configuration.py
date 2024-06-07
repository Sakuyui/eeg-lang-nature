# import torch
# from ..segmentor.abstract_segmentor import *
# from ..machine_learning.models.abstract_prediction_model import *
# from ..machine_learning.models.abstract_segment_preprocessing_model import *
# from ..language.configuration import *
# from ..evaluator.abstract_evaluator import *

# class ExperimentConfiguration(object):
#     def __init__(self, lexer: AbstractSegmenter, prediction_model: AbstractPredictionModel, evaluator, language_configuration: LanguageConfiguration, segment_preprocessing_model: AbstracSegmentPreprocessingModel, feed_segment_increasementally = False):
#         self.segmentor = segmentor
#         self.prediction_model = prediction_model
#         self.evaluator = evaluator
#         self.language_configuration = language_configuration
#         self.segment_preprocessing_model = segment_preprocessing_model
#         self.feed_segment_increasementally = feed_segment_increasementally
    
#     def do_experiment(self, source_signal):
#         # segmenting the signals
#         segmentation_solution = self.segmentor.do_segment(source_signal)
#         with torch.no_grad():
#             if not self.feed_segment_increasementally:
#                 # a model for mapping segmentation_solution to another representation
#                 segment_representations = self.segment_preprocessing_model(signal_info, segmentation_solution.get_segment_endpoints())
#                 outputs = self.prediction_model(segment_representations)
#                 print(self.evaluator.evaluate_outputs(outputs))
#             else:
#                 segment_representations = self.segment_preprocessing_model(segmentation_solution.get_segment_endpoints())
#                 outputs = []
#                 for segment_representation in segment_representations:
#                     output = self.prediction_model(segment_representation)
#                     outputs.append(output)
                    
