# -*- coding: utf-8 -*-
"""
@Time    : 2/15/2024 10:30 AM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a dummy attention layer for illustration purposes
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_features):
        super(AttentionLayer, self).__init__()
        # Define attention layer components here
        # For simplicity, just a linear layer
        self.attention_fc = nn.Linear(input_dim, attention_features)

    def forward(self, obsWact_features, sur_nei_obs):
        # Example attention computation
        # Here we just concatenate and pass through a linear layer
        combined_input = torch.cat((obsWact_features, sur_nei_obs), dim=-1)
        return self.attention_fc(combined_input)


class CustomAgentLayer(nn.Module):
    def __init__(self, critic_obs, n_actions, attention_features):
        super(CustomAgentLayer, self).__init__()
        self.linear = nn.Linear(critic_obs[0] + critic_obs[1] + n_actions, 128)
        self.relu = nn.ReLU()
        self.attention = AttentionLayer(128 + critic_obs[2], attention_features)

    def forward(self, obsWact, sur_nei_obs):
        # Pass through linear and ReLU layers
        x = self.linear(obsWact)
        x = self.relu(x)
        # Combine obsWact and sur_nei_obs for attention
        combined_input = torch.cat((x, sur_nei_obs), dim=-1)
        # Pass through attention layer
        return self.attention(combined_input)

# Define the main critic class
class Critic_ATT_combine_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(Critic_ATT_combine_TwoPortion, self).__init__()
        self.n_agents = n_agents
        # Define individual agent layers including the attention layer
        # self.individual_agent_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(critic_obs[0] + critic_obs[1] + n_actions, 128),
        #         nn.ReLU(),
        #         AttentionLayer(128 + critic_obs[2], 128)  # Adjust the dimension as needed
        #     ) for _ in range(n_agents)
        # ])
        self.individual_agent_layers = nn.ModuleList([
            CustomAgentLayer(critic_obs, n_actions, 128) for _ in range(n_agents)
        ])
        # ... rest of the class ...

    def forward(self, combine_state, combine_action):
        agent_features = []
        for agent_idx in range(self.n_agents):
            agent_obs = torch.cat((combine_state[0][:, agent_idx, :], combine_state[1][:, agent_idx, :]), dim=1)
            if isinstance(combine_action, list):
                agent_act = combine_action[agent_idx]
            else:
                agent_act = combine_action[:, agent_idx, :]
            obsWact = torch.cat((agent_obs, agent_act), dim=1)
            sur_nei_obs = combine_state[2][:, agent_idx, :, :]
            # Pass the tensors to the corresponding module in individual_agent_layers
            single_agent_feature = self.individual_agent_layers[agent_idx](obsWact, sur_nei_obs)
            agent_features.append(single_agent_feature)
        # ... rest of the forward method ...

        return torch.cat(agent_features, dim=1)  # Example return

# Example usage:
# Define the dimensions for the tensors
critic_obs = [20, 20, 7]  # Replace with the actual dimensions
n_agents = 3
n_actions = 5
single_history = 10  # Replace with the actual dimension
hidden_state_size = 128  # Replace with the actual dimension

# Create dummy input data
batch_size = 12
combine_state = [torch.rand(batch_size, n_agents, critic_obs[0]),
                 torch.rand(batch_size, n_agents, critic_obs[1]),
                 torch.rand(batch_size, n_agents, 2, critic_obs[2])]  # Assuming 2 neighbors for simplicity
combine_action = torch.rand(batch_size, n_agents, n_actions)

# Instantiate the Critic_ATT_combine_TwoPortion class
critic_model = Critic_ATT_combine_TwoPortion(critic_obs, n_agents, n_actions, single_history, hidden_state_size)

# Perform a forward pass
output = critic_model(combine_state, combine_action)