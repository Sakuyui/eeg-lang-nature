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
        self.cost_function = nn.BCELoss()
        # Fix the sequential prediction model's parameters.
            # The optimizer will not update the sequential prediction model's parameters.
        # We'd like to just train the syntax related models, and fix the base sequential prediction model's parameters.
            # We mill conduct comparison between base sequential prediction models and syntax-infused model.
        print("Fix base sequential prediction model...")
        self.sequential_model.requires_grad = False 

        
    def forward(self, word_t, eeg_data_t):
        sequential_model_output_hidden = self.sequential_model.forward(eeg_data_t)[1][-1]
        syntax_model_output = self.syntax_model.forward(word_t)

        combined_features = self.combining_model.forward(sequential_model_output_hidden,\
            syntax_model_output)
        prediction_model_output = self.prediction_model(combined_features)
        return prediction_model_output

    def loss(self, dataset):
        eeg_data = torch.Tensor(dataset[0]['eeg_data']).float()
        words = torch.Tensor(dataset[0]['words'])
        labels = torch.Tensor(dataset[0]['labels']).float()
        eeg_data = eeg_data.view(-1, 19)
        words = words.view(-1)
        labels = labels.view(-1, 1)
        retain_sequential_prediction_model_hidden = self.sequential_model.h
        retain_nn_cky_model_records = self.syntax_model.records
        retain_nn_cky_model_windows_beginning_offset = self.syntax_model.windows_beginning_offset
        retain_nn_cky_model_t = self.syntax_model.t
        self.sequential_model.reset_hidden()
        self.syntax_model.reset_global_context()
        
        loss = 0
        times = range(len(words))
        for i in times:
            target = labels[i]
            output = self(words[i], eeg_data[i,:].view(1, -1))
            loss += self.cost_function(output, target.view(-1))
        self.sequential_model.h = retain_sequential_prediction_model_hidden
        self.syntax_model.records = retain_nn_cky_model_records
        self.syntax_model.windows_beginning_offset = retain_nn_cky_model_windows_beginning_offset
        self.syntax_model.t = retain_nn_cky_model_t
        
        print("loss calculation completed.")
        return loss
        
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
            
            
        

