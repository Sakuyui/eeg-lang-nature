import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from scripts.applications.prediction_models.models.train.train import ModelTrainable

class SyntaxInfusedModel(nn.Module, ModelTrainable):
    def __init__(self, sequential_model, syntax_model, combining_model, prediction_model):
        super(SyntaxInfusedModel, self).__init__()
        self.sequential_model = sequential_model
        self.syntax_model = syntax_model
        self.combining_model = combining_model
        self.prediction_model = prediction_model
        
    def forward(self, word_t, eeg_data_t):
        sequential_model_output_hidden = self.sequential_model.forward(eeg_data_t)[1][-1]
        syntax_model_output = self.syntax_model.forward(word_t)

        combined_features = self.combining_model.forward(sequential_model_output_hidden,\
            syntax_model_output)
        prediction_model_output = self.prediction_model(combined_features)
        return prediction_model_output

    def loss(self, word_t, eeg_data_t):
        pass
        
class SimpleConcatCombing(nn.Module):
    # sequential_model_output shape = [batch, N_1], syntax_model_output shape = [batch, N_2]
    def forward(self, sequential_model_output = None, syntax_model_output = None):
        if sequential_model_output is None and syntax_model_output is None:
            return None
        if sequential_model_output is None:
            return syntax_model_output
        if syntax_model_output is None:
            return sequential_model_output
        
        # Make symmetric hold in processing. i.e. the output irrelavant to the orders of two input features.
        concated_feature1 = torch.cat((sequential_model_output, syntax_model_output))
        concated_feature2 = torch.cat((sequential_model_output, syntax_model_output))
        return (concated_feature1 + concated_feature2) / 2
    
class FCPrediction(nn.Module, ModelTrainable):
    def __init__(self, input_size = 200, output_size = 1):
        super(FCPrediction, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.cost_function = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, infused_feature):
        return self.sigmoid(self.fc.forward(infused_feature))
    
    def loss(self, inputs, targets):
        return self.cost_function(self(inputs), targets)
            
            
        

