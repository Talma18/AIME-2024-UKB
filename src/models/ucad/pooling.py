import torch
from torch import nn


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x):
        return x.mean(dim=0), None


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weight = nn.Parameter(torch.randn(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # x is of shape [num_sequences, batch_size, seq_len, hidden_dim]
        attention_scores = torch.matmul(x, self.attention_weight)  # [num_sequences, batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [num_sequences, batch_size, seq_len]

        attention_probs = self.softmax(attention_scores)  # Softmax over the num_sequences dimension
        weighted_sum = (x * attention_probs.unsqueeze(-1)).sum(dim=0)  # [batch_size, seq_len, hidden_dim]

        return weighted_sum, attention_probs
