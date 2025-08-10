import torch.nn as nn


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, device, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x