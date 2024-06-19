import torch
import torch.nn as nn

class SyntaxInfusedModel(nn.Module):
    def __init__(self, sequential_model, syntax_model, combining_model, prediction_model):
        self.sequential_model = sequential_model
        self.syntax_model = syntax_model
        self.combining_model = combining_model
        self.prediction_model = prediction_model
        
    def forward(self, word):
        sequential_model_output = self.sequential_model.forward(word)
        syntax_model_output = self.syntax_model.forward(word)
        combined_features = self.combining_model(sequential_model_output, /
                        syntax_model_output).forward()
        prediction_model_output = self.prediction_model(combined_features)
        return prediction_model_output
    
class SimpleConcatCombing(nn.Module):
    def forward(self, sequential_model_output, syntax_model_output):
        return tensor.concat(sequential_model_output, syntax_model_output)
        