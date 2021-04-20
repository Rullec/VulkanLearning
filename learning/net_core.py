import torch
import torch.nn as nn
import torch.nn.functional as F

class fc_net(nn.Module):

    def __init__(self, input_size, layers, output_size, device, output_relu = True):
        super(fc_net, self).__init__()

        # define layers
        self.device = device
        self.enalbe_output_relu = output_relu
        self.input = nn.Linear(input_size, layers[0])
        self.middle_layers = []
        for j in range(len(layers) - 1):
            self.middle_layers.append(nn.Linear(layers[j], layers[j+1])) 
        self.middle_layers = torch.nn.ModuleList(self.middle_layers)
        self.output = nn.Linear(layers[-1], output_size)

    def forward(self, x):
        x = self.input(x).to(self.device)
        x = F.relu(x).to(self.device)

        for i in self.middle_layers:
            x = i(x).to(self.device)
            x = F.relu(x).to(self.device)


        x = self.output(x).to(self.device)

        x = F.relu(x).to(self.device)

        return x

