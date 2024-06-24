import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class SyntaxInfusedModel(nn.Module):
    def __init__(self, sequential_model, syntax_model, combining_model, prediction_model):
        super(SyntaxInfusedModel, self).__init__()
        self.sequential_model = sequential_model
        self.syntax_model = syntax_model
        self.combining_model = combining_model
        self.prediction_model = prediction_model
        
    def forward(self, word_t, eeg_data_t):
        sequential_model_output = self.sequential_model.forward(eeg_data_t)
        syntax_model_output = self.syntax_model.forward(word_t)
        combined_features = self.combining_model(sequential_model_output,\
                        syntax_model_output).forward()
        prediction_model_output = self.prediction_model(combined_features)
        return prediction_model_output
    
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
        concated_feature1 = torch.cat([sequential_model_output, syntax_model_output])
        concated_feature2 = torch.cat([sequential_model_output, syntax_model_output])
        
        return (concated_feature1 + concated_feature2) / 2
        
class FCPrediction(nn.Module):
    def __init__(self, input_size = 200, output_size = 1):
        super(FCPrediction, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.cost_fn = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, infused_feature):
        return self.sigmoid(self.fc.forward(infused_feature))
    
    def train(self, input_features, targets, epoches = 10, lr = 0.001, batch_size = 4):
        model = self
        batch_start = torch.arange(0, len(input_features), batch_size)
        optimizer = optim.SGD(self.parameters(), lr = lr)
        
        for epoch in range(epoches):
            total_loss = 0.0
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                print(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    batch_features = input_features[start : start + batch_size]
                    batch_targets = targets[start : start + batch_size]
                    
                    # forward pass
                    pred_targets = model(batch_features)
                    loss = self.cost_function(pred_targets, batch_targets)
                    
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # update weights
                    optimizer.step()
                    
                    # print progress
                    acc = (pred_targets.round() == batch_targets).float().mean()
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
                    total_loss += loss
                print(f"total loss = {total_loss}")

        

