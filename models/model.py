import torch
import math
from torch import nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_steps=1000):
        super().__init__()
        pe = torch.zeros(max_steps, embedding_dim)
        position = torch.arange(0, max_steps).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, timestep):
        if len(timestep.shape) == 2:
            timestep = timestep.squeeze(1)
        return self.pe[timestep]


class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, time_emb_dim=1000):
        super().__init__()

        hidden = (time_emb_dim, 1000, 1000, 1000, 1000, 1000)

        self.fc0 = nn.Linear(input_dim, hidden[0])
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden)-1)]
        )

        self.output = nn.Linear(hidden[-1], output_dim)

    def forward(self, x, t):
        x = self.fc0(x.float())
        t_emb = self.time_embedding(t)
        x = x + t_emb

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        return self.output(x)