# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 10:26 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# create input data
x1 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [10, 11, 0, 0, 0]])  # shape (3, 5)
x2 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])  # shape (2, 5)

# create lengths tensor
l1 = torch.tensor([5, 4, 2])  # length of each sequence in x1
l2 = torch.tensor([5, 3])  # length of each sequence in x2

# create batch
batch = nn.utils.rnn.pad_sequence([x1, x2], batch_first=True)

# pack the sequences
packed_seq, _ = nn.utils.rnn.pack_padded_sequence(batch, lengths=2, batch_first=True)

# check the packed sequence
print(packed_seq)

print("done")