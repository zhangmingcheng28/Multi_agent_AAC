# -*- coding: utf-8 -*-
"""
@Time    : 5/4/2023 9:58 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import torch.nn as nn

# Define the linear layers
sum_k = nn.Linear(4, 4, bias=False)
sum_q = nn.Linear(4, 4, bias=False)
sum_v = nn.Linear(4, 4, bias=False)

# Define the input tensors
surrounding_neighbour_state = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=torch.float)  # shape: (2, 1, 4)
agents_state = torch.tensor([[[9, 10, 11, 12]], [[13, 14, 15, 16]]], dtype=torch.float)  # shape: (2, 1, 4)

# Compute the keys, queries, and values
keys = sum_k(surrounding_neighbour_state)  # shape: (2, 1, 4)
queries = sum_q(agents_state)  # shape: (2, 1, 4)
values = sum_v(surrounding_neighbour_state)  # shape: (2, 1, 4)

scores = torch.bmm(keys, queries.transpose(1, 2))  # shape: (2, 1, 1)
alpha = torch.softmax(scores / torch.sqrt((torch.tensor(keys.size(-1)).float())), dim=1)
print(alpha.sum(dim=-1))  # should output tensor([[1.],[1.]])
attention = torch.bmm(alpha, values)
print(attention)
