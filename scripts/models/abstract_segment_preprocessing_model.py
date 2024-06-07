from torch import nn
class AbstracSegmentPreprocessingModel(nn.Module):
    def __init__(self, signal_info):
        self.signal_info = signal_info
    