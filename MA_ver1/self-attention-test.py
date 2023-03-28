# -*- coding: utf-8 -*-
"""
@Time    : 3/23/2023 2:39 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import torch.nn as nn
import random


# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, num_heads):
#         super().__init__()
#
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads
#
#         self.K = nn.Linear(input_dim, input_dim)
#         self.Q = nn.Linear(input_dim, input_dim)
#         self.V = nn.Linear(input_dim, input_dim)
#
#         self.fc = nn.Linear(input_dim, input_dim)
#
#     def forward(self, x):
#         batch_size, feature_dim = x.shape
#
#         # Apply linear transformations
#         K = self.K(x)
#         Q = self.Q(x)
#         V = self.V(x)
#
#         # Compute attention scores
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
#         attention_weights = nn.functional.softmax(scores, dim=-1)
#
#         # Apply attention to values
#         values = torch.matmul(attention_weights, V)
#
#         # Apply final linear transformation
#         #out = self.fc(values)
#         return values


# Example input with feature dimension m

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.K = nn.Linear(input_dim, input_dim)
        self.Q = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, feature_dim = x.shape

        # Apply linear transformations
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        attention_weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        values = torch.matmul(attention_weights, V)

        return values


m = random.choice([4, 5, 3, 6])
input_vector = torch.randn(1, m)

# Initialize self-attention module with input dimension and number of heads
input_dim = m
num_heads = 2
# selfattn = SelfAttention(input_dim, num_heads)  # multi-head attention
selfattn = SelfAttention(input_dim)

# Apply self-attention to input vector
output = selfattn(input_vector)

# Print output shape
print(output.shape)  # Output shape should be [1, m]