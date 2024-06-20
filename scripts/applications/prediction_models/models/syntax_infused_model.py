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
        combined_features = self.combining_model(sequential_model_output,\
                        syntax_model_output).forward()
        prediction_model_output = self.prediction_model(combined_features)
        return prediction_model_output
    
class SimpleConcatCombing(nn.Module):
    # sequential_model_output shape = [batch, N_1], syntax_model_output shape = [batch, N_2]
    def forward(self, sequential_model_output = None, syntax_model_output = None):
        if sequential_model_output == None and syntax_model_output == None:
            return None
        if sequential_model_output == None:
            return syntax_model_output
        if syntax_model_output == None:
            return sequential_model_output
        return torch.cat((sequential_model_output, syntax_model_output), dim = 1)
    
class FCPrediction(nn.Module):
    def __init__(self, input_size = 200, output_size = 1):
        super(FCPrediction, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, infused_feature):
        return self.fc.forward(infused_feature)
    