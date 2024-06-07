import torch.nn as nn
class AbstractPredictionModel(nn.Module):
    def __init__(self):
        pass
    

class AbstractPreEpilepticStatePredictionModel(object):
    def training(self):
        raise NotImplementedError

class TimeSeriesBasedPredictionModel(AbstractPreEpilepticStatePredictionModel):
    pass

class SyntaxTreeInfusedPredictionModel(AbstractPreEpilepticStatePredictionModel):
    pass