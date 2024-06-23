import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EletrodeValueSequentialModel(nn.Module):
    def __init__(self, args):
        self.channels = args['cnt_channels']
        self.rnn = None
    
    def forward(self, x_t):
        return self.rnn(x_t)


class LNNElectrodeValueBasedPredictionModel(nn.Module):
    def __init__(self, ncp_input_size, hidden_size, output_size, sequence_length):
        super(LNNElectrodeValueBasedPredictionModel, self).__init__()

        self.hidden_size = hidden_size
        self.ncp_input_size = ncp_input_size
        self.sequence_length = sequence_length


        ### DESIGNED NCP architecture
        wiring = AutoNCP(hidden_size, output_size)    # 234,034 parameters

        # wiring = NCP(
        #     inter_neurons=13,  # Number of inter neurons
        #     command_neurons=4,  # Number of command neurons
        #     motor_neurons=2,  # Number of motor neurons
        #     sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        #     inter_fanout=2,  # How many outgoing synapses has each inter neuron
        #     recurrent_command_synapses=3,  # Now many recurrent synapses are in the
        #     # command neuron layer
        #     motor_fanin=4,  # How many incomming syanpses has each motor neuron
        # )
        self.rnn = CfC(ncp_input_size, wiring)
        self.reset_hidden()
        #make_wiring_diagram(wiring, "kamada")

        ### Fully connected NCP architecture 
        # self.rnn = CfC(ncp_input_size, hidden_size, proj_size = num_classes, batch_first = True) # input shape -> batch_size, seq len, feature_size (input size)  . Batch_first just means we need that batch dim present
        
    def reset_hidden(self):
        self.h = torch.zeros(1, self.hidden_size).to(device)
        
    def batch_forward(self, sequences):
        h = self.h
        for x in sequences:
            ## RNN MODE
            x = x.view(-1, self.sequence_length, self.ncp_input_size)
            out, h_new = self.rnn(x, h)
     
            out = out[:, -1, :]   # we have 28 outputs since each part of sequence generates an output. for classification, we only want the last one
            h = h_new
        self.h = h
        return out, h_new
    
    def forward(self, sequence):
        h = self.h
        ## RNN MODE
        x = sequence.view(-1, self.sequence_length, self.ncp_input_size)
        out, h_new = self.rnn(x, h)
        out = out[:, -1, :]   # we have 28 outputs since each part of sequence generates an output. for classification, we only want the last one
        h = h_new
        self.h = h
        return out, h_new
    
def make_wiring_diagram(wiring, layout):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = wiring.draw_graph(layout=layout,neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()