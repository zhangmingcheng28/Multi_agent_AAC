# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 10:26 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import random
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn as nn

# Define the input data
x = [
    [[30, 56, 44, 66, 55, 13, 48, 23, 100], [35, 2, 19, 84, 53, 7], [61, 40, 58, 47, 14, 97, 38], [59, 70, 98],
     [39, 20, 38]],
    [[72, 88, 54, 91, 1, 56, 62], [31, 48, 99, 45, 60, 56, 1, 100, 86], [8, 21, 29, 73, 13, 9, 23, 21],
     [64, 36, 60, 54, 85, 1, 77, 24], [19, 15, 72], [94, 56, 58], [94, 82, 56, 24, 29, 63, 75, 63]],
    [[8, 91, 24, 78, 82, 1, 63, 74, 20], [49, 3, 96, 66, 15], [16, 49, 81, 76, 42, 46, 62, 13], [65, 70, 11, 96],
     [38, 38, 92, 34, 78, 21, 50, 43, 10], [57, 54, 66, 45, 23, 95, 62, 88], [82, 5, 92],
     [42, 93, 78, 19, 55, 70, 37, 66, 76, 27]]
]

# # Find the maximum sequence length and feature length
# max_seq_len = max([len(sample) for sample in x])
# max_feature_len = max([len(feature) for sample in x for feature in sample])
#
# # Create tensors for the input data
# x_padded = torch.zeros((len(x), max_seq_len, max_feature_len))
# for i, sample in enumerate(x):
#     for j, feature in enumerate(sample):
#         x_padded[i, j, :len(feature)] = torch.tensor(feature)
#
#
# # Define the model
# class AttentionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(AttentionModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.linear = nn.Linear(hidden_dim * 2, output_dim)
#
#     def forward(self, x):
#         output, (hidden, _) = self.lstm(x)
#         hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
#         output = self.linear(hidden)
#         return output
#
#
# # Define the model input/output dimensions
# input_dim = max_feature_len
# hidden_dim = 10
# output_dim = 5
#
# # Create an instance of the model and pass the input data through it
# model = AttentionModel(input_dim, hidden_dim, output_dim)
# output = model(x_padded)
#
# # Print the output
# print(output)


# class LSTMNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#
#         # Pack variable-length sequences into padded sequences
#         packed_input = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
#
#         # Pass the packed sequence through the LSTM layer
#         packed_output, (hn, cn) = self.lstm(packed_input, (h0, c0))
#
#         # Unpack the padded output sequence
#         padded_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#
#         # Apply a linear layer to each output element
#         output = self.fc(padded_output)
#
#         return output
#
# # Define the input size, hidden size, number of layers, and output size
# input_size = 1
# hidden_size = 64
# num_layers = 2
# output_size = 10
#
# # Initialize the model
# model = LSTMNet(input_size, hidden_size, num_layers, output_size)
#
# # Convert the input to a PyTorch tensor
# x = torch.tensor(x, dtype=torch.float32)
#
# # Pass the input through the model
# output = model(x)
#
# # Print the output shape
# print(output.shape)
import torch
import torch.nn as nn
import numpy as np


def one_hot_encode(state, possible_states):
    num_states = len(possible_states)
    encoding = np.zeros(num_states)
    encoding[possible_states.index(state)] = 1
    return encoding

# Define the possible cell states
possible_states = ['empty', 'object_A', 'object_B', 'object_C']

# Define the flattened grids
grid1 = ['empty', 'object_A', 'object_C', 'object_B', 'empty', 'object_B', 'object_A', 'object_C', 'empty', 'object_A']
grid2 = ['empty', 'object_C', 'empty', 'object_A', 'object_A', 'object_B', 'empty', 'empty', 'object_C', 'object_B', 'object_A']
grid3 = ['object_B', 'empty', 'empty', 'object_A', 'empty', 'object_C', 'object_A', 'empty', 'object_C']

# Encode each grid using one-hot encoding
encoded_grid1 = torch.tensor([one_hot_encode(state, possible_states) for state in grid1])
encoded_grid2 = torch.tensor([one_hot_encode(state, possible_states) for state in grid2])
encoded_grid3 = torch.tensor([one_hot_encode(state, possible_states) for state in grid3])

# Concatenate the encoded grids into a single input tensor
input_tensor = torch.cat((encoded_grid1, encoded_grid2, encoded_grid3), dim=0)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the network hyperparameters
input_size = len(possible_states)
hidden_size = 16
output_size = 2

# Create an instance of the network
net = SimpleNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Pass the input tensor through the network
output = net(input_tensor)

# Print the output tensor
print(output)